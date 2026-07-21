#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH2I.h>
#include <TLatex.h>
#include <TLine.h>
#include <TMath.h>
#include <TRandom.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TTree.h>

#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/ComponentChargedRing.hh"
#include "Garfield/ComponentUser.hh"
#include "Garfield/Medium.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Sensor.hh"

using namespace Garfield;

namespace {

constexpr double kTorrPerBar = 750.061683;
constexpr Long64_t kHardMaxElectronEnergySamples = 200000;
constexpr double kSpaceChargeToleranceCm = 1.0e-5;
constexpr int kGifMaxFrames = 100;
constexpr int kGifDelay = 3;
constexpr int kExcitationCollisionType = 4;
constexpr int kExcitationXYBins = 256;
constexpr int kExcitationZBins = 256;
constexpr int kExcitationTimeBins = 256;
constexpr double kInitialExcitationTimeMaxNs = 10.0;

double quiet_nan() { return std::numeric_limits<double>::quiet_NaN(); }

int parse_int(const char* text, const char* name) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != std::string(text).size()) throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("Invalid integer for ") + name + ": " + text);
  }
}

Long64_t parse_long64(const char* text, const char* name) {
  try {
    std::size_t used = 0;
    const long long value = std::stoll(text, &used);
    if (used != std::string(text).size()) throw std::invalid_argument("trailing characters");
    return static_cast<Long64_t>(value);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("Invalid integer for ") + name + ": " + text);
  }
}

double parse_double(const char* text, const char* name) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != std::string(text).size()) throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("Invalid number for ") + name + ": " + text);
  }
}

struct SimulationConfig {
  std::string root_file;
  double electric_field_v_cm = 0.0;
  double gap_mm = 0.0;
  double pressure_bar = 0.0;
  int npe = 0;
  std::string gas1;
  double composition1 = 0.0;
  std::string gas2;
  double composition2 = 0.0;
  double height = 1.0;
  bool legacy_print_table = false;
  int job_id = 0;

  bool enable_space_charge = false;
  bool make_gif = false;
  Long64_t max_electron_energy_samples = kHardMaxElectronEnergySamples;
  std::string gif_file;

  double temperature_k = 293.15;
  double initial_energy_ev = 0.1;
  double max_electron_energy_ev = 400.0;
  int electron_energy_bins = 1000;

  double pressure_torr() const { return pressure_bar * kTorrPerBar; }
  double gap_cm() const { return gap_mm * 0.1; }

  // Preserve the geometry convention already used by electricUniform.
  double sensor_extent_cm() const { return gap_cm() * height; }
  double component_zmax_cm() const { return sensor_extent_cm() * height; }
};

SimulationConfig parse_arguments(int argc, char* argv[]) {
  if (argc < 13) {
    throw std::runtime_error(
        "Usage: ./uniformE rootFileName.root fieldE(V/cm) gap(mm) pressure(bar) "
        "npe gas1 mixture1(%) gas2 mixture2(%) height printTable jobId "
        "[enableSpaceCharge] [makeGif] [maxEEDInputs] [gifFile]");
  }

  SimulationConfig c;
  c.root_file = argv[1];
  c.electric_field_v_cm = parse_double(argv[2], "fieldE");
  c.gap_mm = parse_double(argv[3], "gap");
  c.pressure_bar = parse_double(argv[4], "pressure");
  c.npe = parse_int(argv[5], "npe");
  c.gas1 = argv[6];
  c.composition1 = parse_double(argv[7], "mixture1");
  c.gas2 = argv[8];
  c.composition2 = parse_double(argv[9], "mixture2");
  c.height = parse_double(argv[10], "height");
  c.legacy_print_table = parse_int(argv[11], "printTable") == 0;
  c.job_id = parse_int(argv[12], "jobId");

  if (argc > 13) c.enable_space_charge = parse_int(argv[13], "enableSpaceCharge") != 0;
  if (argc > 14) c.make_gif = parse_int(argv[14], "makeGif") != 0;
  if (argc > 15) c.max_electron_energy_samples = parse_long64(argv[15], "maxEEDInputs");
  // New CLI: gifFile is argv[16]. For compatibility with the immediately
  // previous version, also accept the old argv[18] position.
  if (argc > 18) {
    c.gif_file = argv[18];
  } else if (argc > 16) {
    c.gif_file = argv[16];
  }

  if (c.electric_field_v_cm <= 0.0) throw std::runtime_error("fieldE must be positive");
  if (c.gap_mm <= 0.0) throw std::runtime_error("gap must be positive");
  if (c.pressure_bar <= 0.0) throw std::runtime_error("pressure must be positive");
  if (c.npe <= 0) throw std::runtime_error("npe must be positive");
  if (c.height <= 0.0) throw std::runtime_error("height must be positive");
  if (c.composition1 < 0.0 || c.composition2 < 0.0) {
    throw std::runtime_error("gas compositions cannot be negative");
  }
  if (c.max_electron_energy_samples < 0 ||
      c.max_electron_energy_samples > kHardMaxElectronEnergySamples) {
    throw std::runtime_error("maxEEDInputs must be between 0 and 200000");
  }
  if (c.make_gif && (c.gif_file.empty() || c.gif_file == "none")) {
    c.gif_file = c.root_file;
    const std::size_t root_pos = c.gif_file.rfind(".root");
    if (root_pos != std::string::npos) c.gif_file.erase(root_pos);
    c.gif_file += "_avalanche.gif";
  }

  return c;
}

struct LevelInfo {
  int gas_index = -1;
  int type = 0;
  double energy_ev = quiet_nan();
  std::string description;
};

std::vector<LevelInfo> read_levels(MediumMagboltz& gas) {
  const int n_levels = gas.GetNumberOfLevels();
  std::vector<LevelInfo> levels;
  levels.reserve(std::max(0, n_levels));
  for (int i = 0; i < n_levels; ++i) {
    LevelInfo info;
    gas.GetLevel(i, info.gas_index, info.type, info.description, info.energy_ev);
    levels.push_back(info);
  }
  return levels;
}

struct ElectronEnergyReservoir {
  Long64_t seen = 0;
  Long64_t maximum = 0;
  std::vector<float> samples;

  void configure(const Long64_t requested_maximum) {
    seen = 0;
    maximum = std::max<Long64_t>(
        0, std::min<Long64_t>(requested_maximum,
                              kHardMaxElectronEnergySamples));
    samples.clear();
    samples.reserve(static_cast<std::size_t>(maximum));
  }

  void observe(const double energy_ev, const bool hole) {
    if (hole || maximum <= 0 || !std::isfinite(energy_ev) || energy_ev < 0.0) return;
    ++seen;
    if (static_cast<Long64_t>(samples.size()) < maximum) {
      samples.push_back(static_cast<float>(energy_ev));
      return;
    }

    const double u = gRandom != nullptr
                         ? gRandom->Rndm()
                         : static_cast<double>(std::rand()) / (RAND_MAX + 1.0);
    const Long64_t index =
        static_cast<Long64_t>(std::floor(u * static_cast<double>(seen)));
    if (index >= 0 && index < maximum) {
      samples[static_cast<std::size_t>(index)] = static_cast<float>(energy_ev);
    }
  }

  double sample_or(const double fallback) const {
    if (samples.empty()) return fallback;
    const std::size_t index = gRandom != nullptr
                                  ? static_cast<std::size_t>(gRandom->Integer(samples.size()))
                                  : static_cast<std::size_t>(std::rand()) % samples.size();
    return static_cast<double>(samples[index]);
  }

  void fill_histogram(TH1D& histogram) const {
    for (const float energy : samples) histogram.Fill(static_cast<double>(energy));
  }
};

ElectronEnergyReservoir g_energy_reservoir;

void user_handle_step(double, double, double, double, double energy,
                      double, double, double, bool hole) {
  g_energy_reservoir.observe(energy, hole);
}

struct ExcitationHistogramRecorder {
  Long64_t excitation_counter = 0;
  TH1D* levels_histogram = nullptr;
  TH2I* xy_histogram = nullptr;
  TH2I* zt_histogram = nullptr;
  bool collect_ion_positions = false;

  // Space charge still needs the ion creation points during the current
  // primary avalanche, but they are never persisted to the ROOT file.
  std::vector<std::array<double, 3>> ion_positions_this_primary;

  void connect(TH1D& levels, TH2I& xy, TH2I& zt,
               const bool keep_ion_positions) {
    levels_histogram = &levels;
    xy_histogram = &xy;
    zt_histogram = &zt;
    collect_ion_positions = keep_ion_positions;
  }

  void begin_primary() { ion_positions_this_primary.clear(); }

  void observe(const double x, const double y, const double z, const double t,
               const int process_type, const int h_level) {
    // Keep ion positions only in memory when space charge is enabled. No ion
    // propagation and no ion-position output are produced.
    if (collect_ion_positions && process_type == 1 && std::isfinite(x) &&
        std::isfinite(y) && std::isfinite(z)) {
      ion_positions_this_primary.push_back({x, y, z});
    }

    // Garfield++ uses collision type 4 for electron-impact excitation.
    if (process_type != kExcitationCollisionType) return;

    ++excitation_counter;
    if (levels_histogram != nullptr && h_level >= 0) {
      levels_histogram->Fill(h_level);
    }
    if (xy_histogram != nullptr && std::isfinite(x) && std::isfinite(y)) {
      xy_histogram->Fill(x, y);
    }
    if (zt_histogram != nullptr && std::isfinite(z) && std::isfinite(t)) {
      zt_histogram->Fill(z, t);
    }
  }
};

ExcitationHistogramRecorder g_excitation_histograms;

void user_handle_collision(double x, double y, double z, double t,
                           int type, int level, Medium*,
                           double, double, double, double, double,
                           double, double, double) {
  g_excitation_histograms.observe(x, y, z, t, type, level);
}

struct SpaceChargeState {
  bool enabled = false;
  Long64_t n_ion_rings = 0;
  std::unique_ptr<ComponentChargedRing> rings;

  void initialise(const SimulationConfig& config, MediumMagboltz& gas) {
    enabled = config.enable_space_charge;
    if (!enabled) return;

    rings = std::make_unique<ComponentChargedRing>();
    const double transverse_extent = 8.0 * config.sensor_extent_cm();
    rings->SetArea(-transverse_extent, -transverse_extent, 0.0,
                   transverse_extent, transverse_extent,
                   config.sensor_extent_cm());
    rings->SetSpacingTolerance(kSpaceChargeToleranceCm);
    rings->SetMedium(&gas);
    rings->ClearActiveRings();
    rings->UpdateCentre(0.0, 0.0);
  }

  void add_current_primary_ions(const SimulationConfig& config) {
    if (!enabled || rings == nullptr) return;
    const double zmax = config.sensor_extent_cm();
    const double xymax = 8.0 * config.sensor_extent_cm();
    for (const auto& position : g_excitation_histograms.ion_positions_this_primary) {
      if (std::abs(position[0]) > xymax || std::abs(position[1]) > xymax ||
          position[2] < 0.0 || position[2] > zmax) {
        continue;
      }
      rings->AddChargedRing(position[0], position[1], position[2], +1.0);
      ++n_ion_rings;
    }
  }
};

struct PrimaryRow {
  Int_t ne = 0;
  Int_t ni = 0;
  Int_t npe = 0;
};

struct EndpointRow {
  Int_t status = 0;
};

struct RunSummary {
  Long64_t ne_total = 0;
  Long64_t ni_total = 0;
  double ne_mean = quiet_nan();
  double ni_mean = quiet_nan();
  double gain_sim = quiet_nan();
  double alpha_eff = quiet_nan();
  double alpha_from_ne = quiet_nan();
  double alpha_from_ni = quiet_nan();
  double vz = quiet_nan();
  bool valid_for_alpha = false;
  std::string alpha_source = "simulation_npe";
};

void write_progress(const int job_id, const int current, const int total) {
  std::cout << "PROGRESS " << job_id << " " << current << " " << total
            << std::endl;
}

std::string gif_frame_filename(const std::string& gif_file,
                               const bool final_frame) {
  return gif_file + (final_frame ? "++" : "+" + std::to_string(kGifDelay));
}

void write_avalanche_gif(const AvalancheMicroscopic& avalanche,
                         const SimulationConfig& config) {
  if (!config.make_gif || config.gif_file.empty()) return;

  struct GifPoint {
    double x = 0.0;
    double z = 0.0;
    double t = 0.0;
  };

  std::vector<std::vector<GifPoint>> tracks;
  double tmin = std::numeric_limits<double>::infinity();
  double tmax = -std::numeric_limits<double>::infinity();

  for (const auto& electron : avalanche.GetElectrons()) {
    if (electron.path.empty()) continue;
    std::vector<GifPoint> track;
    track.reserve(electron.path.size());
    for (const auto& point : electron.path) {
      if (!std::isfinite(point.x) || !std::isfinite(point.z) ||
          !std::isfinite(point.t)) {
        continue;
      }
      track.push_back({point.x, point.z, point.t});
      tmin = std::min(tmin, point.t);
      tmax = std::max(tmax, point.t);
    }
    if (!track.empty()) tracks.push_back(std::move(track));
  }

  if (tracks.empty() || !std::isfinite(tmin) || !std::isfinite(tmax)) {
    std::cerr << "[GIF] No stored drift-line points; GIF was not created.\n";
    return;
  }

  gROOT->SetBatch(kTRUE);
  gSystem->Unlink(config.gif_file.c_str());

  const int nframes = std::max(
      2, std::min(kGifMaxFrames,
                  static_cast<int>(std::ceil(std::max(1.0, 20.0 * (tmax - tmin))))));
  const double xlim = std::max(config.gap_cm(), config.sensor_extent_cm());
  const double zlim = std::max(config.gap_cm(), config.sensor_extent_cm());

  TCanvas canvas("electricUniformGifCanvas", "electricUniform avalanche", 900, 700);
  TH2D frame_axis("electricUniformGifAxes",
                  "Microscopic avalanche;x [cm];z [cm]",
                  10, -xlim, xlim, 10, 0.0, zlim);
  frame_axis.SetStats(false);
  frame_axis.SetDirectory(nullptr);
  TLatex label;
  label.SetNDC(true);
  label.SetTextSize(0.04);

  for (int iframe = 0; iframe < nframes; ++iframe) {
    const double fraction = nframes > 1
                                ? static_cast<double>(iframe) / (nframes - 1)
                                : 1.0;
    const double frame_time = tmin + fraction * (tmax - tmin);

    canvas.cd();
    canvas.Clear();
    frame_axis.Draw("AXIS");

    TLine anode(-xlim, 0.0, xlim, 0.0);
    TLine cathode(-xlim, config.gap_cm(), xlim, config.gap_cm());
    anode.Draw("SAME");
    cathode.Draw("SAME");

    std::vector<std::unique_ptr<TGraph>> graphs;
    graphs.reserve(tracks.size());
    for (const auto& track : tracks) {
      auto graph = std::make_unique<TGraph>();
      int point_index = 0;
      for (const auto& point : track) {
        if (point.t > frame_time) break;
        graph->SetPoint(point_index++, point.x, point.z);
      }
      if (point_index > 0) graph->Draw("L SAME");
      graphs.push_back(std::move(graph));
    }

    char time_text[128];
    std::snprintf(time_text, sizeof(time_text), "#it{t} = %.5g ns", frame_time);
    label.DrawLatex(0.17, 0.90, time_text);

    canvas.Modified();
    canvas.Update();
    canvas.Print(gif_frame_filename(config.gif_file, iframe + 1 == nframes).c_str());
  }

  std::cout << "[GIF] Guardado en: " << config.gif_file << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const std::time_t wall_start = std::time(nullptr);
    const std::clock_t cpu_start = std::clock();

    SimulationConfig config = parse_arguments(argc, argv);

    std::cout << std::setprecision(8)
              << "SetComposition: " << config.gas1 << "/" << config.gas2
              << " = " << config.composition1 << "/" << config.composition2 << "\n"
              << "[CONFIG] spaceCharge=" << (config.enable_space_charge ? 1 : 0)
              << ", makeGif=" << (config.make_gif ? 1 : 0)
              << ", maxEEDInputs=" << config.max_electron_energy_samples << "\n"
              << "[CONFIG] excitation output = hExcXY + hExcZT ("
              << kExcitationXYBins << "x" << kExcitationXYBins << ", "
              << kExcitationZBins << "x" << kExcitationTimeBins << ")\n";

    auto gas = std::make_unique<MediumMagboltz>();
    if (config.composition2 <= 0.0) {
      gas->SetComposition(config.gas1, config.composition1);
    } else {
      gas->SetComposition(config.gas1, config.composition1,
                          config.gas2, config.composition2);
    }
    gas->SetTemperature(config.temperature_k);
    gas->SetPressure(config.pressure_torr());
    gas->SetMaxElectronEnergy(config.max_electron_energy_ev);
    gas->EnableDebugging();
    gas->PrintGas();
    gas->Initialise();
    gas->DisableDebugging();

    const std::vector<LevelInfo> levels = read_levels(*gas);
    std::cout << "# of levels: " << levels.size() << "\n";

    auto field_component = std::make_unique<ComponentUser>();
    const double uniform_e = config.electric_field_v_cm;
    field_component->SetElectricField(
        [uniform_e](double, double, double, double& ex, double& ey, double& ez) {
          ex = 0.0;
          ey = 0.0;
          ez = uniform_e;
        });
    field_component->SetArea(-8.0 * config.sensor_extent_cm(),
                             -8.0 * config.sensor_extent_cm(), 0.0,
                             8.0 * config.sensor_extent_cm(),
                             8.0 * config.sensor_extent_cm(),
                             config.component_zmax_cm());
    field_component->SetMedium(gas.get());

    SpaceChargeState space_charge;
    space_charge.initialise(config, *gas);

    auto sensor = std::make_unique<Sensor>();
    sensor->AddComponent(field_component.get());
    if (space_charge.enabled && space_charge.rings != nullptr) {
      sensor->AddComponent(space_charge.rings.get());
    }
    sensor->SetArea(-8.0 * config.sensor_extent_cm(),
                    -8.0 * config.sensor_extent_cm(), 0.0,
                    8.0 * config.sensor_extent_cm(),
                    8.0 * config.sensor_extent_cm(),
                    config.sensor_extent_cm());

    TFile output(config.root_file.c_str(), "RECREATE");
    if (output.IsZombie()) {
      throw std::runtime_error("Could not create ROOT file: " + config.root_file);
    }

    TH1D h_electron_energy(
        "hElectronEnergyDistribution",
        "Electron energy distribution from null-collision steps;E_{e} [eV];samples",
        config.electron_energy_bins, 0.0, 50.0);
    TH1D h_levels("hLevels", "Electron-impact excitation levels;hLevel;excitations",
                  std::max(1, static_cast<int>(levels.size())), 0.0,
                  static_cast<double>(std::max(1, static_cast<int>(levels.size()))));

    const double transverse_extent = 8.0 * config.sensor_extent_cm();
    TH2I h_exc_xy(
        "hExcXY", "Excitation transverse distribution;x [cm];y [cm]",
        kExcitationXYBins, -transverse_extent, transverse_extent,
        kExcitationXYBins, -transverse_extent, transverse_extent);
    TH2I h_exc_zt(
        "hExcZT", "Excitation longitudinal-time distribution;z [cm];t [ns]",
        kExcitationZBins, 0.0, config.sensor_extent_cm(),
        kExcitationTimeBins, 0.0, kInitialExcitationTimeMaxNs);
    // The number of bins stays fixed. ROOT enlarges the time range only if an
    // excitation falls outside it, keeping file size bounded.
    h_exc_zt.SetCanExtend(TH1::kYaxis);

    TTree data_per_primary("dataPerPrimaryElectron",
                           "Data per primary electron");
    TTree data_per_electron("dataPerElectron", "Data per electron endpoint");
    TTree gas_data("gasData", "Gas configuration and avalanche summary");

    PrimaryRow primary_row;
    data_per_primary.Branch("ne", &primary_row.ne, "ne/I");
    data_per_primary.Branch("ni", &primary_row.ni, "ni/I");
    data_per_primary.Branch("npe", &primary_row.npe, "npe/I");

    EndpointRow endpoint_row;
    data_per_electron.Branch("status", &endpoint_row.status, "status/I");

    g_excitation_histograms.connect(
        h_levels, h_exc_xy, h_exc_zt, config.enable_space_charge);
    g_energy_reservoir.configure(config.max_electron_energy_samples);

    RunSummary summary;
    double pressure_torr = config.pressure_torr();
    int npe_for_tree = config.npe;
    Long64_t eed_samples_seen = 0;
    Long64_t eed_samples_stored = 0;
    Long64_t n_excitations = 0;
    bool space_charge_enabled = config.enable_space_charge;
    bool gif_enabled = config.make_gif;

    gas_data.Branch("electricField", &config.electric_field_v_cm, "electricField/D");
    gas_data.Branch("gap_mm", &config.gap_mm, "gap_mm/D");
    gas_data.Branch("pressure", &pressure_torr, "pressure/D");
    gas_data.Branch("pressureBar", &config.pressure_bar, "pressureBar/D");
    gas_data.Branch("temp", &config.temperature_k, "temp/D");
    gas_data.Branch("gas1", &config.gas1);
    gas_data.Branch("composition1", &config.composition1, "composition1/D");
    gas_data.Branch("gas2", &config.gas2);
    gas_data.Branch("composition2", &config.composition2, "composition2/D");
    gas_data.Branch("height", &config.height, "height/D");
    gas_data.Branch("npe", &npe_for_tree, "npe/I");
    gas_data.Branch("neTotal", &summary.ne_total, "neTotal/L");
    gas_data.Branch("niTotal", &summary.ni_total, "niTotal/L");
    gas_data.Branch("neMean", &summary.ne_mean, "neMean/D");
    gas_data.Branch("niMean", &summary.ni_mean, "niMean/D");
    gas_data.Branch("gainSim", &summary.gain_sim, "gainSim/D");
    gas_data.Branch("alphaEff", &summary.alpha_eff, "alphaEff/D");
    gas_data.Branch("alphaFromNe", &summary.alpha_from_ne, "alphaFromNe/D");
    gas_data.Branch("alphaFromNi", &summary.alpha_from_ni, "alphaFromNi/D");
    gas_data.Branch("vz", &summary.vz, "vz/D");
    gas_data.Branch("validForAlpha", &summary.valid_for_alpha, "validForAlpha/O");
    gas_data.Branch("alphaSource", &summary.alpha_source);
    gas_data.Branch("spaceChargeEnabled", &space_charge_enabled,
                    "spaceChargeEnabled/O");
    gas_data.Branch("nSpaceChargeIons", &space_charge.n_ion_rings,
                    "nSpaceChargeIons/L");
    gas_data.Branch("gifEnabled", &gif_enabled, "gifEnabled/O");
    gas_data.Branch("electronEnergySamplesSeen", &eed_samples_seen,
                    "electronEnergySamplesSeen/L");
    gas_data.Branch("electronEnergySamplesStored", &eed_samples_stored,
                    "electronEnergySamplesStored/L");
    gas_data.Branch("nExcitations", &n_excitations, "nExcitations/L");

    auto avalanche = std::make_unique<AvalancheMicroscopic>();
    avalanche->SetSensor(sensor.get());
    avalanche->EnableSignalCalculation(false);
    avalanche->EnableNullCollisionSteps(true, 1);
    avalanche->SetUserHandleStep(user_handle_step);
    avalanche->SetUserHandleCollision(user_handle_collision);
    if (config.make_gif) avalanche->EnableDriftLines(true);

    for (int event_number = 0; event_number < config.npe; ++event_number) {
      g_excitation_histograms.begin_primary();

      double x0 = gRandom->Uniform(-config.sensor_extent_cm(),
                                   config.sensor_extent_cm());
      double y0 = gRandom->Uniform(-config.sensor_extent_cm(),
                                   config.sensor_extent_cm());
      const double z0 = config.gap_cm();
      const double t0 = 0.0;
      const double e0 = event_number == 0
                            ? config.initial_energy_ev
                            : g_energy_reservoir.sample_or(config.initial_energy_ev);

      const double phi0 = gRandom->Uniform(0.0, 2.0 * TMath::Pi());
      // Preserve the original electricUniform launch direction (3 pi / 4).
      const double theta0 = 0.75 * TMath::Pi();
      const double dx0 = TMath::Cos(phi0) * TMath::Sin(theta0);
      const double dy0 = TMath::Sin(phi0) * TMath::Sin(theta0);
      const double dz0 = TMath::Cos(theta0);

      if (config.make_gif && event_number == 0) {
        x0 = 0.0;
        y0 = 0.0;
      }

      avalanche->AvalancheElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);
      avalanche->GetAvalancheSize(primary_row.ne, primary_row.ni);
      primary_row.npe = 1;
      summary.ne_total += primary_row.ne;
      summary.ni_total += primary_row.ni;
      data_per_primary.Fill();

      for (int electron = 0; electron < primary_row.ne; ++electron) {
        double ex0 = 0.0, ey0 = 0.0, ez0 = 0.0, et0 = 0.0, ee0 = 0.0;
        double ex1 = 0.0, ey1 = 0.0, ez1 = 0.0, et1 = 0.0, ee1 = 0.0;
        avalanche->GetElectronEndpoint(electron,
                                       ex0, ey0, ez0, et0, ee0,
                                       ex1, ey1, ez1, et1, ee1,
                                       endpoint_row.status);
        data_per_electron.Fill();
      }

      // The new ions affect subsequent primary avalanches, not the avalanche in
      // which they were created. No ion propagation is performed or written.
      space_charge.add_current_primary_ions(config);

      if (config.make_gif && event_number == 0) {
        write_avalanche_gif(*avalanche, config);
      }

      write_progress(config.job_id, event_number + 1, config.npe);
    }

    if (config.npe > 0) {
      summary.ne_mean = static_cast<double>(summary.ne_total) / config.npe;
      summary.ni_mean = static_cast<double>(summary.ni_total) / config.npe;
      summary.gain_sim = summary.ne_mean;
    }

    const double gap_cm = config.gap_cm();
    if (std::isfinite(summary.gain_sim) && summary.gain_sim > 0.0 && gap_cm > 0.0) {
      summary.alpha_from_ne = std::log(summary.gain_sim) / gap_cm;
      summary.alpha_eff = summary.alpha_from_ne;
    }
    if (std::isfinite(summary.ni_mean) && gap_cm > 0.0) {
      summary.alpha_from_ni = summary.ni_mean / gap_cm;
    }

    summary.valid_for_alpha =
        config.npe > 100 && std::isfinite(summary.alpha_eff) && summary.alpha_eff > 0.0;
    if (!summary.valid_for_alpha) {
      summary.alpha_eff = quiet_nan();
      summary.alpha_from_ne = quiet_nan();
      summary.alpha_from_ni = quiet_nan();
    }

    g_energy_reservoir.fill_histogram(h_electron_energy);
    eed_samples_seen = g_energy_reservoir.seen;
    eed_samples_stored =
        static_cast<Long64_t>(g_energy_reservoir.samples.size());
    n_excitations = g_excitation_histograms.excitation_counter;

    gas_data.Fill();

    output.cd();
    h_electron_energy.Write("hElectronEnergyDistribution", TObject::kOverwrite);
    h_levels.Write("hLevels", TObject::kOverwrite);
    h_exc_xy.Write("hExcXY", TObject::kOverwrite);
    h_exc_zt.Write("hExcZT", TObject::kOverwrite);
    data_per_primary.Write("dataPerPrimaryElectron", TObject::kOverwrite);
    data_per_electron.Write("dataPerElectron", TObject::kOverwrite);
    gas_data.Write("gasData", TObject::kOverwrite);
    output.Close();

    std::cout << "average # of electrons produced: " << summary.ne_mean << "\n"
              << "average # of ions produced: " << summary.ni_mean << "\n"
              << "simulated alphaEff: " << summary.alpha_eff
              << " 1/cm (validForAlpha=" << (summary.valid_for_alpha ? 1 : 0) << ")\n"
              << "EED samples stored/seen: " << eed_samples_stored << "/"
              << eed_samples_seen << "\n"
              << "Excitations accumulated: " << n_excitations << "\n"
              << "space-charge ion rings: " << space_charge.n_ion_rings << "\n";

    const double wall_time = std::difftime(std::time(nullptr), wall_start);
    const double cpu_time =
        static_cast<double>(std::clock() - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "It took you " << wall_time << " seconds to finish.\n"
              << "CPU time = " << cpu_time << "s\n"
              << "DONE " << config.job_id << "\n"
              << "Finalizando correctamente uniformE().\n";

    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "[ERROR] " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
