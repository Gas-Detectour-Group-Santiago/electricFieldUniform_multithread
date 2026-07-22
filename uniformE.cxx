#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH2I.h>
#include <TLatex.h>
#include <TLine.h>
#include <TMath.h>
#include <TPad.h>
#include <TRandom.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>

#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/ComponentChargedRing.hh"
#include "Garfield/ComponentUser.hh"
#include "Garfield/Medium.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/ViewDrift.hh"

using namespace Garfield;

namespace {

constexpr double kTorrPerBar = 750.061683;
constexpr int kElasticType = 0;
constexpr int kIonisationType = 1;
constexpr int kAttachmentType = 2;
constexpr int kInelasticType = 3;
constexpr int kExcitationType = 4;
constexpr int kSuperelasticType = 5;
constexpr int kEnergyBins = 1000;
constexpr int kExcitationBins = 256;
constexpr Long64_t kMaxEnergySamples = 200000;
constexpr double kInitialTimeRangeNs = 10.0;
constexpr double kSpaceChargeToleranceCm = 1.0e-5;

// ============================================================================
//                              Configuration
// ============================================================================

int parse_int(const char* text, const char* name) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != std::string(text).size()) throw std::invalid_argument("suffix");
    return value;
  } catch (...) {
    throw std::runtime_error(std::string("Invalid integer for ") + name + ": " + text);
  }
}

double parse_double(const char* text, const char* name) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != std::string(text).size()) throw std::invalid_argument("suffix");
    return value;
  } catch (...) {
    throw std::runtime_error(std::string("Invalid number for ") + name + ": " + text);
  }
}

struct Config {
  std::string root_file;
  std::string mixture_name;
  double field_v_cm = 0.0;
  double gap_mm = 0.0;
  double pressure_bar = 0.0;
  int min_npe = 10;
  int max_npe = 100;
  double target_relative_error = 0.03;
  std::string gas1;
  double composition1 = 0.0;
  std::string gas2;
  double composition2 = 0.0;
  double height_factor = 1.5;
  bool space_charge = false;
  bool make_gif = false;
  double gif_tmax_ns = 0.0;
  int gif_frames = 80;
  bool gif_move_ions = true;
  double gif_ion_speed_cm_ns = 1.0e-4;
  bool record_excitation_positions = true;
  bool measure_gas_transport = true;
  int job_id = 0;
  std::string gif_file;

  double temperature_k = 293.15;
  double initial_energy_ev = 0.1;
  double max_electron_energy_ev = 400.0;

  double pressure_torr() const { return pressure_bar * kTorrPerBar; }
  double gap_cm() const { return 0.1 * gap_mm; }
  double launch_z_cm() const { return gap_cm(); }
  double z_max_cm() const { return height_factor * gap_cm(); }
  double xy_half_width_cm() const { return 2.0 * gap_cm(); }
};

struct LevelInfo {
  int level = -1;
  int gas_index = -1;
  int component_slot = -1;
  int process_type = -1;
  std::string gas_name;
  std::string description;
  double energy_ev = std::numeric_limits<double>::quiet_NaN();
};

std::vector<LevelInfo> read_level_info(const Config& config,
                                       MediumMagboltz& gas) {
  const int n_levels =
      std::max(0, static_cast<int>(gas.GetNumberOfLevels()));

  // MediumMagboltz::GetLevel returns ngas as the index of the gas in the
  // active mixture. Components with a zero fraction are not passed to
  // SetComposition, so map the active Magboltz index back to the original
  // gas1/gas2 slot used in the configuration.
  std::vector<int> active_slots;
  if (config.composition1 > 0.0) active_slots.push_back(0);
  if (config.composition2 > 0.0) active_slots.push_back(1);

  std::vector<LevelInfo> levels;
  levels.reserve(static_cast<std::size_t>(n_levels));

  for (int level = 0; level < n_levels; ++level) {
    LevelInfo info;
    info.level = level;
    gas.GetLevel(static_cast<unsigned int>(level), info.gas_index,
                 info.process_type, info.description, info.energy_ev);

    // Garfield++ stores ngas as a zero-based index in the active mixture
    // (m_csType / nCsTypes). Convert it back to the original gas1/gas2 slot.
    if (info.gas_index >= 0 &&
        info.gas_index < static_cast<int>(active_slots.size())) {
      info.component_slot =
          active_slots[static_cast<std::size_t>(info.gas_index)];
      info.gas_name = info.component_slot == 0 ? config.gas1 : config.gas2;
    }
    levels.push_back(std::move(info));
  }
  return levels;
}

Config read_config(int argc, char* argv[]) {
  if (argc < 19) {
    throw std::runtime_error(
        "Usage: ./uniformE output.root mixture field(V/cm) gap(mm) pressure(bar) "
        "minNpe maxNpe targetRelativeError gas1 comp1 gas2 comp2 heightFactor "
        "spaceCharge makeGif gifTmax(ns) gifFrames jobId [gifFile] "
        "[gifMoveIons] [gifIonSpeedCmNs] [recordExcitationPositions] "
        "[measureGasTransport]");
  }

  Config c;
  c.root_file = argv[1];
  c.mixture_name = argv[2];
  c.field_v_cm = parse_double(argv[3], "field");
  c.gap_mm = parse_double(argv[4], "gap");
  c.pressure_bar = parse_double(argv[5], "pressure");
  c.min_npe = parse_int(argv[6], "minNpe");
  c.max_npe = parse_int(argv[7], "maxNpe");
  c.target_relative_error = parse_double(argv[8], "targetRelativeError");
  c.gas1 = argv[9];
  c.composition1 = parse_double(argv[10], "composition1");
  c.gas2 = argv[11];
  c.composition2 = parse_double(argv[12], "composition2");
  c.height_factor = parse_double(argv[13], "heightFactor");
  c.space_charge = parse_int(argv[14], "spaceCharge") != 0;
  c.make_gif = parse_int(argv[15], "makeGif") != 0;
  c.gif_tmax_ns = parse_double(argv[16], "gifTmax");
  c.gif_frames = parse_int(argv[17], "gifFrames");
  c.job_id = parse_int(argv[18], "jobId");
  if (argc > 19) c.gif_file = argv[19];
  if (argc > 20) c.gif_move_ions = parse_int(argv[20], "gifMoveIons") != 0;
  if (argc > 21) c.gif_ion_speed_cm_ns = parse_double(argv[21], "gifIonSpeedCmNs");
  if (argc > 22) {
    c.record_excitation_positions =
        parse_int(argv[22], "recordExcitationPositions") != 0;
  }
  if (argc > 23) {
    c.measure_gas_transport =
        parse_int(argv[23], "measureGasTransport") != 0;
  }

  if (c.field_v_cm <= 0.0) throw std::runtime_error("field must be positive");
  if (c.gap_mm <= 0.0) throw std::runtime_error("gap must be positive");
  if (c.pressure_bar <= 0.0) throw std::runtime_error("pressure must be positive");
  if (c.min_npe <= 0 || c.max_npe < c.min_npe) {
    throw std::runtime_error("Require 0 < minNpe <= maxNpe");
  }
  if (c.target_relative_error < 0.0) {
    throw std::runtime_error("targetRelativeError cannot be negative");
  }
  if (c.height_factor < 1.0) {
    throw std::runtime_error("heightFactor must be at least 1");
  }
  if (c.composition1 < 0.0 || c.composition2 < 0.0) {
    throw std::runtime_error("Gas compositions cannot be negative");
  }
  if (c.composition1 <= 0.0 && c.composition2 <= 0.0) {
    throw std::runtime_error("At least one gas composition must be positive");
  }
  if (c.gif_ion_speed_cm_ns < 0.0) {
    throw std::runtime_error("gifIonSpeedCmNs cannot be negative");
  }
  if (c.make_gif && c.gif_file.empty()) {
    c.gif_file = c.root_file + ".gif";
  }
  c.gif_frames = std::max(2, std::min(500, c.gif_frames));
  return c;
}

// ============================================================================
//                    Electron-energy distribution
// ============================================================================

struct EnergyReservoir {
  Long64_t seen = 0;
  std::vector<float> values;

  void reset() {
    seen = 0;
    values.clear();
    values.reserve(static_cast<std::size_t>(kMaxEnergySamples));
  }

  void add(const double energy_ev, const bool hole) {
    if (hole || !std::isfinite(energy_ev) || energy_ev < 0.0) return;
    ++seen;

    if (static_cast<Long64_t>(values.size()) < kMaxEnergySamples) {
      values.push_back(static_cast<float>(energy_ev));
      return;
    }

    const Long64_t index = static_cast<Long64_t>(
        std::floor(gRandom->Rndm() * static_cast<double>(seen)));
    if (index >= 0 && index < kMaxEnergySamples) {
      values[static_cast<std::size_t>(index)] = static_cast<float>(energy_ev);
    }
  }

  double random_energy(const double fallback) const {
    if (values.empty()) return fallback;
    return values[static_cast<std::size_t>(gRandom->Integer(values.size()))];
  }
};

EnergyReservoir g_energy;

void handle_step(double, double, double, double, double energy,
                 double, double, double, bool hole) {
  g_energy.add(energy, hole);
}

// ============================================================================
//                  Excitation and ionisation callbacks
// ============================================================================

struct CollisionRecorder {
  // Preserve the historical compact output: hLevels stores every real
  // non-elastic collision term (type != 0). No redundant per-gas or per-type
  // level histograms are written.
  TH1D* h_levels = nullptr;
  TH2I* h_xy = nullptr;
  TH2I* h_zt = nullptr;
  const std::vector<LevelInfo>* level_info = nullptr;
  bool keep_ions = false;

  Long64_t n_non_elastic = 0;
  Long64_t n_non_elastic_gas1 = 0;
  Long64_t n_non_elastic_gas2 = 0;
  Long64_t n_inelastic_type3 = 0;
  Long64_t n_inelastic_type3_gas1 = 0;
  Long64_t n_inelastic_type3_gas2 = 0;
  Long64_t n_excitation_type4 = 0;
  Long64_t n_excitation_type4_gas1 = 0;
  Long64_t n_excitation_type4_gas2 = 0;
  Long64_t n_excitation_like = 0;
  Long64_t n_excitation_like_gas1 = 0;
  Long64_t n_excitation_like_gas2 = 0;
  Long64_t n_unassigned_levels = 0;
  Long64_t n_ionisations = 0;
  Long64_t n_attachments = 0;
  Long64_t n_superelastic = 0;
  std::vector<std::array<double, 4>> ions_this_primary;

  void reset(TH1D& levels,
             const std::vector<LevelInfo>& levels_info,
             TH2I* xy, TH2I* zt, const bool store_ions) {
    h_levels = &levels;
    h_xy = xy;
    h_zt = zt;
    level_info = &levels_info;
    keep_ions = store_ions;

    n_non_elastic = 0;
    n_non_elastic_gas1 = 0;
    n_non_elastic_gas2 = 0;
    n_inelastic_type3 = 0;
    n_inelastic_type3_gas1 = 0;
    n_inelastic_type3_gas2 = 0;
    n_excitation_type4 = 0;
    n_excitation_type4_gas1 = 0;
    n_excitation_type4_gas2 = 0;
    n_excitation_like = 0;
    n_excitation_like_gas1 = 0;
    n_excitation_like_gas2 = 0;
    n_unassigned_levels = 0;
    n_ionisations = 0;
    n_attachments = 0;
    n_superelastic = 0;
    ions_this_primary.clear();
  }

  void start_primary() { ions_this_primary.clear(); }

  int component_slot_for_level(const int level) const {
    if (level < 0 || level_info == nullptr ||
        level >= static_cast<int>(level_info->size())) {
      return -1;
    }
    return (*level_info)[static_cast<std::size_t>(level)].component_slot;
  }

  void add(double x, double y, double z, double t, int type, int level) {
    if (type == kIonisationType) {
      ++n_ionisations;
      if (keep_ions && std::isfinite(x) && std::isfinite(y) &&
          std::isfinite(z)) {
        ions_this_primary.push_back({x, y, z, t});
      }
    } else if (type == kAttachmentType) {
      ++n_attachments;
    } else if (type == kSuperelasticType) {
      ++n_superelastic;
    }

    const int component_slot = component_slot_for_level(level);
    const bool has_level = level >= 0;

    // Historical hLevels: every non-elastic real collision with a valid
    // cross-section term. This intentionally includes ionisation, attachment,
    // generic inelastic, excitation and super-elastic channels.
    if (type != kElasticType && has_level) {
      ++n_non_elastic;
      if (h_levels != nullptr) h_levels->Fill(level);
      if (component_slot == 0) {
        ++n_non_elastic_gas1;
      } else if (component_slot == 1) {
        ++n_non_elastic_gas2;
      } else {
        ++n_unassigned_levels;
      }
    }

    if (type == kInelasticType && has_level) {
      ++n_inelastic_type3;
      if (component_slot == 0) {
        ++n_inelastic_type3_gas1;
      } else if (component_slot == 1) {
        ++n_inelastic_type3_gas2;
      }
    }

    if (type == kExcitationType && has_level) {
      ++n_excitation_type4;
      if (component_slot == 0) {
        ++n_excitation_type4_gas1;
      } else if (component_slot == 1) {
        ++n_excitation_type4_gas2;
      }
    }

    // Positions used by the photon/scintillation stage include both generic
    // inelastic molecular channels and discrete excitation channels. This is
    // essential for CF4, whose relevant channels are not all classified as 4.
    if ((type == kInelasticType || type == kExcitationType) && has_level) {
      ++n_excitation_like;
      if (component_slot == 0) {
        ++n_excitation_like_gas1;
      } else if (component_slot == 1) {
        ++n_excitation_like_gas2;
      }
      if (h_xy != nullptr && std::isfinite(x) && std::isfinite(y)) {
        h_xy->Fill(x, y);
      }
      if (h_zt != nullptr && std::isfinite(z) && std::isfinite(t)) {
        h_zt->Fill(z, t);
      }
    }
  }
};

CollisionRecorder g_collisions;

void handle_collision(double x, double y, double z, double t,
                      int type, int level, Medium*,
                      double, double, double, double, double,
                      double, double, double) {
  g_collisions.add(x, y, z, t, type, level);
}

// ============================================================================
//                           Running statistics
// ============================================================================

struct RunningStatistics {
  int n = 0;
  double mean = 0.0;
  double m2 = 0.0;

  void add(const double value) {
    ++n;
    const double delta = value - mean;
    mean += delta / n;
    m2 += delta * (value - mean);
  }

  double standard_deviation() const {
    return n > 1 ? std::sqrt(m2 / (n - 1)) : 0.0;
  }

  double error_on_mean() const {
    return n > 1 ? standard_deviation() / std::sqrt(static_cast<double>(n))
                 : std::numeric_limits<double>::infinity();
  }

  double relative_error() const {
    return mean > 0.0 ? error_on_mean() / mean
                      : std::numeric_limits<double>::infinity();
  }
};

// ============================================================================
//                         Magboltz gas transport
// ============================================================================

struct GasTransport {
  bool enabled = false;
  bool has_velocity = false;
  bool has_diffusion = false;
  bool has_townsend = false;
  bool has_attachment = false;
  double vx_cm_ns = std::numeric_limits<double>::quiet_NaN();
  double vy_cm_ns = std::numeric_limits<double>::quiet_NaN();
  double vz_cm_ns = std::numeric_limits<double>::quiet_NaN();
  double drift_speed_cm_ns = std::numeric_limits<double>::quiet_NaN();
  double longitudinal_diffusion = std::numeric_limits<double>::quiet_NaN();
  double transverse_diffusion = std::numeric_limits<double>::quiet_NaN();
  double townsend_cm_inv = std::numeric_limits<double>::quiet_NaN();
  double attachment_cm_inv = std::numeric_limits<double>::quiet_NaN();

  void measure(const Config& config, MediumMagboltz& gas) {
    enabled = config.measure_gas_transport;
    if (!enabled) return;

    // Initialise() prepares the microscopic collision table used by
    // AvalancheMicroscopic, but it does not generate the macroscopic transport
    // table.  Build a tiny three-point grid centred on the exact simulation
    // field so ElectronVelocity/Townsend/Attachment have real Magboltz data.
    const double half_width = std::max(1.0, 1.0e-3 * config.field_v_cm);
    const double field_min = std::max(1.0e-6, config.field_v_cm - half_width);
    const double field_max = config.field_v_cm + half_width;
    gas.SetFieldGrid(field_min, field_max, 3, false);
    std::cout << "[Magboltz] transport table around E = "
              << config.field_v_cm << " V/cm" << std::endl;
    gas.GenerateGasTable(3);

    const double ex = 0.0;
    const double ey = 0.0;
    const double ez = config.field_v_cm;
    const double bx = 0.0;
    const double by = 0.0;
    const double bz = 0.0;

    has_velocity = gas.ElectronVelocity(
        ex, ey, ez, bx, by, bz, vx_cm_ns, vy_cm_ns, vz_cm_ns);
    if (has_velocity) {
      drift_speed_cm_ns = std::sqrt(
          vx_cm_ns * vx_cm_ns + vy_cm_ns * vy_cm_ns + vz_cm_ns * vz_cm_ns);
    }

    has_diffusion = gas.ElectronDiffusion(
        ex, ey, ez, bx, by, bz, longitudinal_diffusion, transverse_diffusion);
    has_townsend = gas.ElectronTownsend(
        ex, ey, ez, bx, by, bz, townsend_cm_inv);
    has_attachment = gas.ElectronAttachment(
        ex, ey, ez, bx, by, bz, attachment_cm_inv);

    if (!has_velocity || !has_diffusion || !has_townsend || !has_attachment) {
      throw std::runtime_error(
          "Magboltz transport table was generated, but velocity/diffusion/"
          "Townsend/attachment could not be read at the simulation field.");
    }
  }
};

// ============================================================================
//                              Space charge
// ============================================================================

struct SpaceCharge {
  std::unique_ptr<ComponentChargedRing> rings;
  Long64_t n_ions = 0;

  void initialise(const Config& config, MediumMagboltz& gas) {
    if (!config.space_charge) return;

    rings = std::make_unique<ComponentChargedRing>();
    rings->SetArea(-config.xy_half_width_cm(), -config.xy_half_width_cm(), 0.0,
                   config.xy_half_width_cm(), config.xy_half_width_cm(),
                   config.z_max_cm());
    rings->SetSpacingTolerance(kSpaceChargeToleranceCm);
    rings->SetMedium(&gas);
    rings->ClearActiveRings();
    rings->UpdateCentre(0.0, 0.0);
  }

  void add_new_ions(const Config& config) {
    if (rings == nullptr) return;

    for (const auto& ion : g_collisions.ions_this_primary) {
      if (std::abs(ion[0]) > config.xy_half_width_cm() ||
          std::abs(ion[1]) > config.xy_half_width_cm() ||
          ion[2] < 0.0 || ion[2] > config.z_max_cm()) {
        continue;
      }
      rings->AddChargedRing(ion[0], ion[1], ion[2], +1.0);
      ++n_ions;
    }
  }
};

// ============================================================================
//                                  GIF
// ============================================================================

struct GifIon {
  double x = 0.0;
  double y = 0.0;
  double z0 = 0.0;
  double t0 = 0.0;
};

class GifPlotter {
 public:
  GifPlotter(const Config& config, Sensor& sensor)
      : config_(config), sensor_(sensor),
        canvas_("cGif", "electricUniform avalanche", 1400, 600) {
    gStyle->SetPalette(kBird);
    gStyle->SetNumberContours(100);

    canvas_.Divide(2, 1);
    drift_view_.SetElectronsToFront();
    drift_view_.SetPlane(0.0, -1.0, 0.0, 0.0, 0.0, 0.0);
    drift_view_.SetArea(-config_.xy_half_width_cm(), 0.0,
                         config_.xy_half_width_cm(), config_.gap_cm());
    drift_view_.SetCanvas(static_cast<TPad*>(canvas_.cd(1)));

    label_.SetTextSize(0.040);
    label_.SetTextFont(42);
  }

  void connect(AvalancheMicroscopic& avalanche) {
    avalanche.EnableExcitationMarkers(false);
    avalanche.EnableIonisationMarkers(false);
    avalanche.EnableAttachmentMarkers(false);
    avalanche.EnablePlotting(&drift_view_);
  }

  void clear() { drift_view_.Clear(); }

  void draw_frame(const int frame, const double time_ns,
                  const SpaceCharge& space_charge,
                  const std::vector<std::array<double, 3>>& ion_positions,
                  const bool last_frame) {
    draw_drift_panel(frame, time_ns, ion_positions);
    draw_field_panel(time_ns, space_charge);

    canvas_.Modified();
    canvas_.Update();
    gSystem->ProcessEvents();

    const std::string suffix = last_frame ? "++" : "+3";
    canvas_.Print((config_.gif_file + suffix).c_str());
  }

 private:
  void prepare_pad(TPad* pad, const bool colour_bar) {
    if (pad == nullptr) return;
    pad->SetLeftMargin(0.14);
    pad->SetBottomMargin(0.13);
    pad->SetTopMargin(0.06);
    pad->SetRightMargin(colour_bar ? 0.20 : 0.05);
    pad->SetGridx(true);
    pad->SetGridy(true);
    pad->SetTicks(1, 1);
  }

  void draw_drift_panel(const int frame, const double time_ns,
                        const std::vector<std::array<double, 3>>& ion_positions) {
    TPad* pad = static_cast<TPad*>(canvas_.cd(1));
    prepare_pad(pad, false);

    drift_view_.SetArea(-config_.xy_half_width_cm(), 0.0,
                         config_.xy_half_width_cm(), config_.gap_cm());
    drift_view_.Plot2d(true, true);

    ion_markers_.reset();
    if (!ion_positions.empty()) {
      ion_markers_ = std::make_unique<TGraph>();
      int point = 0;
      for (const auto& ion : ion_positions) {
        if (std::abs(ion[0]) > config_.xy_half_width_cm()) continue;
        if (ion[2] < 0.0 || ion[2] > config_.gap_cm()) continue;
        ion_markers_->SetPoint(point++, ion[0], ion[2]);
      }
      if (point > 0) {
        ion_markers_->SetMarkerStyle(20);
        ion_markers_->SetMarkerSize(0.45);
        ion_markers_->SetMarkerColor(kBlue + 2);
        ion_markers_->Draw("P SAME");
      }
    }

    char text[128];
    std::snprintf(text, sizeof(text), "#it{t} = %.4g ns", time_ns);
    label_.DrawLatexNDC(0.06, 0.90, text);
    std::snprintf(text, sizeof(text), "Frame %d", frame);
    label_.DrawLatexNDC(0.06, 0.82, text);
  }

  void draw_field_panel(const double time_ns, const SpaceCharge& space_charge) {
    TPad* pad = static_cast<TPad*>(canvas_.cd(2));
    pad->Clear();
    prepare_pad(pad, true);

    constexpr int n_x = 160;
    constexpr int n_z = 160;
    field_map_ = std::make_unique<TH2D>(
        "hGifField", "", n_x,
        -config_.xy_half_width_cm(), config_.xy_half_width_cm(),
        n_z, 0.0, config_.gap_cm());
    field_map_->SetDirectory(nullptr);
    field_map_->SetStats(false);
    field_map_->GetXaxis()->SetTitle("x [cm]");
    field_map_->GetYaxis()->SetTitle("z [cm]");

    const bool show_space_charge =
        config_.space_charge && space_charge.rings != nullptr &&
        space_charge.rings->GetNumberOfRings() > 0;

    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

    for (int ix = 1; ix <= n_x; ++ix) {
      const double x = field_map_->GetXaxis()->GetBinCenter(ix);

      if (!show_space_charge) {
        for (int iz = 1; iz <= n_z; ++iz) {
          const double z = field_map_->GetYaxis()->GetBinCenter(iz);
          const double value = -config_.field_v_cm * z;
          field_map_->SetBinContent(ix, iz, value);
          minimum = std::min(minimum, value);
          maximum = std::max(maximum, value);
        }
        continue;
      }

      // ComponentChargedRing provides the electric field reliably, but its
      // potential output is not suitable for this live map. Reconstruct the
      // space-charge potential from the longitudinal field instead:
      //
      //   Delta V_SC(x,z) = - integral_0^z E_z,SC(x,z') dz'.
      //
      // E is in V/cm and z in cm, so the integral is directly in volts.
      auto sample_ez_sc = [&](const double z) {
        double ex = 0.0;
        double ey = 0.0;
        double ez = 0.0;
        double potential = 0.0;
        int status = 0;
        Medium* medium = nullptr;
        space_charge.rings->ElectricField(
            x, 0.0, z, ex, ey, ez, potential, medium, status);
        return std::isfinite(ez) ? ez : 0.0;
      };

      double z_previous = 0.0;
      double ez_previous = sample_ez_sc(z_previous);
      double delta_v = 0.0;

      for (int iz = 1; iz <= n_z; ++iz) {
        const double z = field_map_->GetYaxis()->GetBinCenter(iz);
        const double ez = sample_ez_sc(z);
        delta_v += -0.5 * (ez_previous + ez) * (z - z_previous);

        field_map_->SetBinContent(ix, iz, delta_v);
        minimum = std::min(minimum, delta_v);
        maximum = std::max(maximum, delta_v);

        z_previous = z;
        ez_previous = ez;
      }
    }

    if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
        std::abs(maximum - minimum) < 1.0e-15) {
      const double scale = show_space_charge ? 1.0e-12 : 1.0;
      minimum = -scale;
      maximum = scale;
    }

    if (show_space_charge) {
      const double symmetric = std::max(std::abs(minimum), std::abs(maximum));
      field_map_->SetMinimum(-symmetric);
      field_map_->SetMaximum(+symmetric);
      field_map_->GetZaxis()->SetTitle("#Delta V_{SC}(z,t) [V]");
    } else {
      field_map_->SetMinimum(-config_.field_v_cm * config_.gap_cm());
      field_map_->SetMaximum(0.0);
      field_map_->GetZaxis()->SetTitle("V_{0}(z) [V]");
    }

    field_map_->GetZaxis()->SetTitleOffset(1.25);
    field_map_->SetContour(100);
    field_map_->Draw("COLZ");

    char text[128];
    std::snprintf(text, sizeof(text), "#it{t} = %.4g ns", time_ns);
    label_.DrawLatexNDC(0.06, 0.90, text);
    label_.DrawLatexNDC(
        0.06, 0.82,
        show_space_charge ? "#Delta V_{SC}(x,z) [V]" : "V_{0}(z) [V]");
  }

  const Config& config_;
  Sensor& sensor_;
  TCanvas canvas_;
  ViewDrift drift_view_;
  TLatex label_;
  std::unique_ptr<TH2D> field_map_;
  std::unique_ptr<TGraph> ion_markers_;
};

void rebuild_live_space_charge(
    const Config& config, SpaceCharge& space_charge,
    const AvalancheMicroscopic& avalanche,
    const std::vector<std::array<double, 3>>& ion_positions) {
  if (space_charge.rings == nullptr) return;

  space_charge.rings->ClearActiveRings();
  space_charge.rings->UpdateCentre(0.0, 0.0);
  space_charge.n_ions = 0;

  // Positive ions use their current visual positions.
  for (const auto& ion : ion_positions) {
    if (std::abs(ion[0]) > config.xy_half_width_cm() ||
        std::abs(ion[1]) > config.xy_half_width_cm() ||
        ion[2] < 0.0 || ion[2] > config.gap_cm()) {
      continue;
    }
    space_charge.rings->AddChargedRing(ion[0], ion[1], ion[2], +1.0);
    ++space_charge.n_ions;
  }

  // Electrons still inside the multiplication gap contribute negative charge.
  for (const auto& electron : avalanche.GetElectrons()) {
    if (electron.path.empty()) continue;
    const auto& point = electron.path.back();
    if (std::abs(point.x) > config.xy_half_width_cm() ||
        std::abs(point.y) > config.xy_half_width_cm() ||
        point.z <= kSpaceChargeToleranceCm ||
        point.z >= config.gap_cm() - kSpaceChargeToleranceCm) {
      continue;
    }
    space_charge.rings->AddChargedRing(point.x, point.y, point.z, -1.0);
  }
}

void run_live_gif(AvalancheMicroscopic& avalanche, Sensor& sensor,
                  const Config& config, SpaceCharge& space_charge,
                  const double x0, const double y0, const double z0,
                  const double t0, const double e0,
                  const double dx0, const double dy0, const double dz0) {
  if (!config.make_gif || config.gif_file.empty()) return;

  gSystem->Unlink(config.gif_file.c_str());

  GifPlotter plotter(config, sensor);
  plotter.connect(avalanche);
  avalanche.AddElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);

  const double tmax = std::max(1.0e-9, config.gif_tmax_ns);
  const double dt = tmax / std::max(1, config.gif_frames);
  std::size_t processed_ions = 0;
  std::vector<GifIon> ions;

  for (int frame = 0; frame < config.gif_frames; ++frame) {
    const double time_min = frame * dt;
    const double time_max = (frame + 1) * dt;

    plotter.clear();
    avalanche.SetTimeWindow(time_min, time_max);
    avalanche.ResumeAvalanche();

    while (processed_ions < g_collisions.ions_this_primary.size()) {
      const auto& ion = g_collisions.ions_this_primary[processed_ions++];
      ions.push_back({ion[0], ion[1], ion[2], ion[3]});
    }

    std::vector<std::array<double, 3>> ion_positions;
    ion_positions.reserve(ions.size());
    for (const GifIon& ion : ions) {
      double z = ion.z0;
      if (config.gif_move_ions && time_max > ion.t0) {
        z += config.gif_ion_speed_cm_ns * (time_max - ion.t0);
      }
      z = std::clamp(z, 0.0, config.gap_cm());
      ion_positions.push_back({ion.x, ion.y, z});
    }

    rebuild_live_space_charge(config, space_charge, avalanche, ion_positions);
    plotter.draw_frame(frame, time_max, space_charge, ion_positions,
                       frame + 1 == config.gif_frames);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const std::time_t wall_start = std::time(nullptr);
    const std::clock_t cpu_start = std::clock();
    Config config = read_config(argc, argv);

    std::unique_ptr<TApplication> application;
    if (config.make_gif) {
      int root_argc = 1;
      char application_name[] = "uniformE";
      char* root_argv[] = {application_name, nullptr};
      application = std::make_unique<TApplication>(
          "electricUniform", &root_argc, root_argv);
      gROOT->SetBatch(kFALSE);
    }

    // ========================================================================
    //                              Setup gas
    // ========================================================================

    MediumMagboltz gas;
    if (config.composition1 > 0.0 && config.composition2 > 0.0) {
      gas.SetComposition(config.gas1, config.composition1,
                         config.gas2, config.composition2);
    } else if (config.composition1 > 0.0) {
      gas.SetComposition(config.gas1, config.composition1);
    } else {
      gas.SetComposition(config.gas2, config.composition2);
    }

    gas.SetTemperature(config.temperature_k);
    gas.SetPressure(config.pressure_torr());
    gas.SetMaxElectronEnergy(config.max_electron_energy_ev);
    gas.Initialise();

    GasTransport gas_transport;
    gas_transport.measure(config, gas);

    const std::vector<LevelInfo> level_info = read_level_info(config, gas);
    const int n_levels =
        std::max(1, static_cast<int>(level_info.size()));

    // ========================================================================
    //                   Geometry and uniform electric field
    // ========================================================================

    // The physical multiplication distance is always gap.
    // The electron starts at z = gap and drifts towards the anode at z = 0.
    // heightFactor only adds free space above the launch plane.

    ComponentUser field;
    const double uniform_field = config.field_v_cm;
    field.SetElectricField(
        [uniform_field](double, double, double,
                        double& ex, double& ey, double& ez) {
          ex = 0.0;
          ey = 0.0;
          ez = uniform_field;
        });
    field.SetArea(-config.xy_half_width_cm(), -config.xy_half_width_cm(), 0.0,
                  config.xy_half_width_cm(), config.xy_half_width_cm(),
                  config.z_max_cm());
    field.SetMedium(&gas);

    SpaceCharge space_charge;
    space_charge.initialise(config, gas);

    Sensor sensor;
    sensor.AddComponent(&field);
    if (space_charge.rings != nullptr) sensor.AddComponent(space_charge.rings.get());
    sensor.SetArea(-config.xy_half_width_cm(), -config.xy_half_width_cm(), 0.0,
                   config.xy_half_width_cm(), config.xy_half_width_cm(),
                   config.z_max_cm());

    // ========================================================================
    //                              ROOT output
    // ========================================================================

    const std::filesystem::path output_path(config.root_file);
    if (!output_path.parent_path().empty()) {
      std::filesystem::create_directories(output_path.parent_path());
    }

    TFile output(config.root_file.c_str(), "RECREATE");
    if (output.IsZombie()) {
      throw std::runtime_error("Could not create ROOT file: " + config.root_file);
    }

    TH1D h_electron_energy(
        "hElectronEnergyDistribution",
        "Electron energy distribution from null-collision steps;E_{e} [eV];samples",
        kEnergyBins, 0.0, 50.0);
    // hLevels keeps the historical meaning used by the previous pipeline:
    // every real non-elastic collision term (types 1--5). This is the only
    // level histogram written to the ROOT file.
    TH1D h_levels(
        "hLevels", "Excitation Distribution;hLevel;excitations",
        n_levels, 0.0, static_cast<double>(n_levels));
    std::unique_ptr<TH2I> h_exc_xy;
    std::unique_ptr<TH2I> h_exc_zt;
    if (config.record_excitation_positions) {
      h_exc_xy = std::make_unique<TH2I>(
          "hExcXY", "Inelastic/excitation transverse distribution;x [cm];y [cm]",
          kExcitationBins, -config.xy_half_width_cm(), config.xy_half_width_cm(),
          kExcitationBins, -config.xy_half_width_cm(), config.xy_half_width_cm());
      h_exc_zt = std::make_unique<TH2I>(
          "hExcZT", "Inelastic/excitation longitudinal-time distribution;z [cm];t [ns]",
          kExcitationBins, 0.0, config.z_max_cm(),
          kExcitationBins, 0.0, kInitialTimeRangeNs);
      h_exc_zt->SetCanExtend(TH1::kYaxis);
    }

    TTree data_per_primary("dataPerPrimaryElectron", "Data per primary electron");
    TTree data_per_electron("dataPerElectron", "Data per electron endpoint");
    TTree gas_data("gasData", "Gas configuration and avalanche summary");
    Int_t ne = 0;
    Int_t ni = 0;
    Int_t one_primary = 1;
    Int_t endpoint_status = 0;

    data_per_primary.Branch("ne", &ne, "ne/I");
    data_per_primary.Branch("ni", &ni, "ni/I");
    data_per_primary.Branch("npe", &one_primary, "npe/I");
    data_per_electron.Branch("status", &endpoint_status, "status/I");

    // ========================================================================
    //                         Microscopic avalanche
    // ========================================================================

    g_energy.reset();
    g_collisions.reset(
        h_levels, level_info, h_exc_xy.get(), h_exc_zt.get(),
        config.space_charge || config.make_gif);

    AvalancheMicroscopic avalanche;
    avalanche.SetSensor(&sensor);
    avalanche.EnableSignalCalculation(false);
    avalanche.EnableNullCollisionSteps(true, 1);
    avalanche.SetUserHandleStep(handle_step);
    avalanche.SetUserHandleCollision(handle_collision);
    if (config.make_gif) avalanche.EnableDriftLines(true);

    RunningStatistics electron_statistics;
    RunningStatistics ion_statistics;
    Long64_t ne_total = 0;
    Long64_t ni_total = 0;

    for (int event = 0; event < config.max_npe; ++event) {
      g_collisions.start_primary();

      double x0 = gRandom->Uniform(-config.gap_cm(), config.gap_cm());
      double y0 = gRandom->Uniform(-config.gap_cm(), config.gap_cm());
      const double z0 = config.launch_z_cm();
      const double t0 = 0.0;
      const double e0 = event == 0
                            ? config.initial_energy_ev
                            : g_energy.random_energy(config.initial_energy_ev);

      if (config.make_gif && event == 0) {
        x0 = 0.0;
        y0 = 0.0;
      }

      const double phi = gRandom->Uniform(0.0, 2.0 * TMath::Pi());
      const double theta = 0.75 * TMath::Pi();
      const double dx = TMath::Cos(phi) * TMath::Sin(theta);
      const double dy = TMath::Sin(phi) * TMath::Sin(theta);
      const double dz = TMath::Cos(theta);

      if (config.make_gif && event == 0) {
        run_live_gif(avalanche, sensor, config, space_charge,
                     x0, y0, z0, t0, e0, dx, dy, dz);
      } else {
        avalanche.AvalancheElectron(x0, y0, z0, t0, e0, dx, dy, dz);
      }
      avalanche.GetAvalancheSize(ne, ni);

      ne_total += ne;
      ni_total += ni;
      electron_statistics.add(ne);
      ion_statistics.add(ni);
      data_per_primary.Fill();

      for (int electron = 0; electron < ne; ++electron) {
        double ex0 = 0.0, ey0 = 0.0, ez0 = 0.0, et0 = 0.0, ee0 = 0.0;
        double ex1 = 0.0, ey1 = 0.0, ez1 = 0.0, et1 = 0.0, ee1 = 0.0;
        avalanche.GetElectronEndpoint(
            electron, ex0, ey0, ez0, et0, ee0,
            ex1, ey1, ez1, et1, ee1, endpoint_status);
        data_per_electron.Fill();
      }

      // The live GIF already rebuilt the charged-ring component frame by
      // frame for the first primary. Avoid adding the same ions twice.
      if (!(config.make_gif && event == 0)) {
        space_charge.add_new_ions(config);
      }

      const int completed = event + 1;
      const int progress_step = std::max(1, config.max_npe / 50);
      if (completed == 1 || completed == config.max_npe ||
          completed % progress_step == 0) {
        std::cout << "PROGRESS " << config.job_id << " " << completed << " "
                  << config.max_npe << std::endl;
      }

      const bool enough_primaries = completed >= config.min_npe;
      const bool precision_reached =
          config.target_relative_error > 0.0 && enough_primaries &&
          electron_statistics.relative_error() <= config.target_relative_error;
      if (precision_reached) break;
    }

    // ========================================================================
    //                       Gain and effective alpha
    // ========================================================================

    int actual_npe = electron_statistics.n;
    double gain = electron_statistics.mean;
    double gain_error = electron_statistics.error_on_mean();
    double ni_mean = ion_statistics.mean;
    const double gap_cm = config.gap_cm();

    double alpha_effective = std::numeric_limits<double>::quiet_NaN();
    double alpha_error = std::numeric_limits<double>::quiet_NaN();
    bool valid_for_alpha = false;

    if (gain > 1.0 && gap_cm > 0.0) {
      alpha_effective = std::log(gain) / gap_cm;
      if (std::isfinite(gain_error)) {
        alpha_error = gain_error / (gain * gap_cm);
      }
      valid_for_alpha = std::isfinite(alpha_effective) && alpha_effective > 0.0;
    }

    for (const float energy : g_energy.values) h_electron_energy.Fill(energy);

    double pressure_torr = config.pressure_torr();
    double height_mm = 10.0 * config.z_max_cm();
    double launch_z_mm = 10.0 * config.launch_z_cm();
    double relative_gain_error = gain > 0.0 ? gain_error / gain
                                            : std::numeric_limits<double>::quiet_NaN();
    Long64_t eed_seen = g_energy.seen;
    Long64_t eed_stored = static_cast<Long64_t>(g_energy.values.size());
    // Preserve the historical hLevels count while also exposing the
    // individual Garfield collision categories explicitly.
    Long64_t n_non_elastic = g_collisions.n_non_elastic;
    Long64_t n_non_elastic_gas1 = g_collisions.n_non_elastic_gas1;
    Long64_t n_non_elastic_gas2 = g_collisions.n_non_elastic_gas2;
    Long64_t n_inelastic_type3 = g_collisions.n_inelastic_type3;
    Long64_t n_inelastic_type3_gas1 = g_collisions.n_inelastic_type3_gas1;
    Long64_t n_inelastic_type3_gas2 = g_collisions.n_inelastic_type3_gas2;
    Long64_t n_excitation_type4 = g_collisions.n_excitation_type4;
    Long64_t n_excitation_type4_gas1 = g_collisions.n_excitation_type4_gas1;
    Long64_t n_excitation_type4_gas2 = g_collisions.n_excitation_type4_gas2;
    Long64_t n_excitations = g_collisions.n_excitation_like;
    Long64_t n_excitations_gas1 = g_collisions.n_excitation_like_gas1;
    Long64_t n_excitations_gas2 = g_collisions.n_excitation_like_gas2;
    Long64_t n_unassigned_levels = g_collisions.n_unassigned_levels;
    Long64_t n_ionisations = g_collisions.n_ionisations;
    Long64_t n_attachments = g_collisions.n_attachments;
    Long64_t n_superelastic = g_collisions.n_superelastic;
    bool precision_reached =
        config.target_relative_error > 0.0 &&
        std::isfinite(relative_gain_error) &&
        relative_gain_error <= config.target_relative_error;

    // Compact gasData. Keep only configuration and optional MediumMagboltz
    // transport quantities requested by the campaign.
    double magboltz_alpha = gas_transport.townsend_cm_inv;
    double magboltz_eta = gas_transport.attachment_cm_inv;
    double magboltz_alpha_eff = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(magboltz_alpha) && std::isfinite(magboltz_eta)) {
      magboltz_alpha_eff = magboltz_alpha - magboltz_eta;
    }
    double magboltz_vz_drift = gas_transport.vz_cm_ns;
    double magboltz_longitudinal_diffusion =
        gas_transport.longitudinal_diffusion;
    double magboltz_transverse_diffusion =
        gas_transport.transverse_diffusion;

    gas_data.Branch("gas1", &config.gas1);
    gas_data.Branch(
        "composition1_pct", &config.composition1, "composition1_pct/D");
    gas_data.Branch("gas2", &config.gas2);
    gas_data.Branch(
        "composition2_pct", &config.composition2, "composition2_pct/D");
    gas_data.Branch(
        "pressure_bar", &config.pressure_bar, "pressure_bar/D");
    gas_data.Branch(
        "temperature_K", &config.temperature_k, "temperature_K/D");
    gas_data.Branch(
        "electricField_V_cm", &config.field_v_cm, "electricField_V_cm/D");
    gas_data.Branch("gap_mm", &config.gap_mm, "gap_mm/D");
    gas_data.Branch("height_mm", &height_mm, "height_mm/D");
    gas_data.Branch("spaceCharge", &config.space_charge, "spaceCharge/O");
    gas_data.Branch("npe", &actual_npe, "npe/I");
    gas_data.Branch(
        "townsendAlpha_cm_inv", &magboltz_alpha,
        "townsendAlpha_cm_inv/D");
    gas_data.Branch(
        "attachmentEta_cm_inv", &magboltz_eta,
        "attachmentEta_cm_inv/D");
    gas_data.Branch(
        "alphaEffective_cm_inv", &magboltz_alpha_eff,
        "alphaEffective_cm_inv/D");
    gas_data.Branch(
        "driftVelocityZ_cm_ns", &magboltz_vz_drift,
        "driftVelocityZ_cm_ns/D");
    gas_data.Branch(
        "longitudinalDiffusion_sqrt_cm", &magboltz_longitudinal_diffusion,
        "longitudinalDiffusion_sqrt_cm/D");
    gas_data.Branch(
        "transverseDiffusion_sqrt_cm", &magboltz_transverse_diffusion,
        "transverseDiffusion_sqrt_cm/D");
    gas_data.Fill();

    // ========================================================================
    //                              Save ROOT
    // ========================================================================

    output.cd();
    h_electron_energy.Write("hElectronEnergyDistribution", TObject::kOverwrite);
    h_levels.Write("hLevels", TObject::kOverwrite);
    if (h_exc_xy != nullptr) {
      h_exc_xy->Write("hExcXY", TObject::kOverwrite);
    }
    if (h_exc_zt != nullptr) {
      h_exc_zt->Write("hExcZT", TObject::kOverwrite);
    }
    data_per_primary.Write("dataPerPrimaryElectron", TObject::kOverwrite);
    data_per_electron.Write("dataPerElectron", TObject::kOverwrite);
    gas_data.Write("gasData", TObject::kOverwrite);

    // ROOT automatically attaches histograms and trees created while a TFile is
    // the current directory. These objects are also owned by stack variables or
    // std::unique_ptr in this function. Detach them before closing the file so
    // TFile::Close does not delete them and leave their C++ owners with dangling
    // pointers (which caused the TH2I double-delete crash, exit code 139).
    h_electron_energy.SetDirectory(nullptr);
    h_levels.SetDirectory(nullptr);
    if (h_exc_xy != nullptr) h_exc_xy->SetDirectory(nullptr);
    if (h_exc_zt != nullptr) h_exc_zt->SetDirectory(nullptr);
    data_per_primary.SetDirectory(nullptr);
    data_per_electron.SetDirectory(nullptr);
    gas_data.SetDirectory(nullptr);

    output.Close();

    const double wall_time = std::difftime(std::time(nullptr), wall_start);
    const double cpu_time =
        static_cast<double>(std::clock() - cpu_start) / CLOCKS_PER_SEC;

    std::cout << std::setprecision(8)
              << "RESULT gain=" << gain
              << " gainError=" << gain_error
              << " alphaEffective=" << alpha_effective
              << " alphaError=" << alpha_error
              << " npe=" << actual_npe << "\n"
              << "Non-elastic collisions (hLevels): " << n_non_elastic << "\n"
              << "Inelastic type 3: " << n_inelastic_type3 << "\n"
              << "Excitation type 4: " << n_excitation_type4 << "\n"
              << "Excitation-like type 3+4: " << n_excitations << "\n"
              << "Ionisations: " << n_ionisations << "\n"
              << "Wall time: " << wall_time << " s\n"
              << "CPU time: " << cpu_time << " s\n"
              << "DONE " << config.job_id << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
