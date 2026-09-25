import argparse
import sys
import time

from backend import config
from backend.signal_processing import RealTimeFilter, PanTompkinsQRSDetector, BeatSegmenter
from backend.edge_infer import EdgeInferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description="LiveGuard Edge Pipeline")
    parser.add_argument(
        "--source",
        choices=["ARDUINO", "MOCK", "RPI_SPI", "ADS1115", "CLINICAL"],
        default=config.DATA_SOURCE,
        help="Input data stream source (default: %(default)s)"
    )
    parser.add_argument(
        "--patient",
        type=str,
        default="100",
        help="Patient ID for CLINICAL mode (e.g., 100 [Normal], 119 [Ventricular], 207 [Flutter])"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial COM port for Arduino (e.g., COM3, /dev/ttyACM0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_ANOMALY_THRESHOLD,
        help="Anomaly classification threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--no-ws",
        action="store_true",
        help="Disable WebSocket telemetry streaming"
    )
    return parser.parse_args()


def get_driver(args):
    source = args.source
    if source == "MOCK":
        from backend.drivers.mock_driver import MockSensorDriver
        return MockSensorDriver()

    elif source == "CLINICAL":
        from backend.drivers.clinical_driver import ClinicalPatientDriver
        try:
            return ClinicalPatientDriver(patient_id=args.patient)
        except FileNotFoundError as e:
            print(f"[ERROR] Could not load clinical data for patient #{args.patient}: {e}")
            sys.exit(1)

    elif source == "ARDUINO":
        try:
            from backend.drivers.serial_driver import ArduinoSerialDriver
        except ImportError:
            print("[ERROR] pyserial is required for ARDUINO mode. Run: sudo apt install -y python3-serial")
            sys.exit(1)
        driver = ArduinoSerialDriver(port=args.port)
        driver.connect()
        return driver

    elif source == "RPI_SPI":
        try:
            from backend.drivers.rpi_mcp3008_driver import RPiMCP3008Driver
            return RPiMCP3008Driver()
        except Exception as e:
            print(f"[ERROR] Could not initialize RPi MCP3008 driver: {e}")
            sys.exit(1)

    elif source == "ADS1115":
        try:
            from backend.drivers.ads1115_driver import ADS1115Driver
            return ADS1115Driver()
        except Exception as e:
            print(f"[ERROR] Could not initialize ADS1115 driver: {e}")
            sys.exit(1)

    else:
        raise ValueError(f"Unknown data source: {source}")


def get_telemetry_server(no_ws: bool):
    if no_ws:
        return None
    try:
        from backend.telemetry_server import TelemetryServer
        ws_server = TelemetryServer()
        ws_server.start()
        return ws_server
    except ImportError:
        print("[NOTICE] websockets package not installed. Running in standalone console mode.")
        return None


def main():
    args = parse_args()

    print("=" * 65)
    print(" LiveGuard-EHMS Edge Intelligence Pipeline")
    print(f" Source: {args.source} | Rate: {config.SAMPLING_RATE_ECG} Hz | Window: {config.BEAT_WINDOW_SIZE}")
    if args.source == "CLINICAL":
        print(f" Replaying Hospital Patient: #{args.patient}")
    print("=" * 65)

    driver = get_driver(args)
    filter_engine = RealTimeFilter(fs=config.SAMPLING_RATE_ECG)
    qrs_detector = PanTompkinsQRSDetector(fs=config.SAMPLING_RATE_ECG)
    segmenter = BeatSegmenter(
        pre_r=config.PRE_R_SAMPLES,
        post_r=config.POST_R_SAMPLES,
        window_size=config.BEAT_WINDOW_SIZE
    )
    try:
        infer_engine = EdgeInferenceEngine(threshold=args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Could not initialize inference engine: {e}")
        sys.exit(1)

    ws_server = get_telemetry_server(args.no_ws)

    sample_count = 0
    total_beats = 0
    anomalies_detected = 0
    start_time = time.time()
    print("\n[Status] Pipeline active. Monitoring heartbeats in real-time...\n")

    consecutive_errors = 0
    last_pipeline_error_log = None
    max_consecutive_errors = 50

    try:
        while True:
            try:
                sample_data = driver.read_sample()
                if sample_data is None:
                    consecutive_errors = 0
                    continue

                sample_count += 1
                raw_ecg = sample_data["ecg_raw"]
                leads_off = sample_data["leads_off"]

                if leads_off:
                    print("\r[WARNING] Leads-off detected!", end="", flush=True)
                    consecutive_errors = 0
                    continue

                filtered_ecg = filter_engine.process_sample(raw_ecg)
                is_r_peak = qrs_detector.process_sample(filtered_ecg)
                current_hr = qrs_detector.get_heart_rate()

                beat_tensor = segmenter.add_sample(filtered_ecg, is_r_peak)
                prediction_info = None

                if beat_tensor is not None:
                    total_beats += 1
                    prediction_info = infer_engine.predict_beat(beat_tensor)

                    if prediction_info["is_anomaly"]:
                        anomalies_detected += 1
                        status_str = f"[ABNORMAL/ARRHYTHMIA] (Prob: {prediction_info['abnormal_prob']:.2f})"
                    else:
                        status_str = f"[NORMAL BEAT] (Conf: {prediction_info['confidence']}%)"

                    print(
                        f"Beat #{total_beats:04d} | HR: {current_hr:.1f} BPM | {status_str} "
                        f"| Total Alerts: {anomalies_detected}",
                        flush=True
                    )

                    if ws_server:
                        ws_server.broadcast({
                            "type": "beat",
                            "prediction": prediction_info["prediction"],
                            "is_anomaly": prediction_info["is_anomaly"],
                            "confidence": prediction_info["confidence"],
                            "abnormal_prob": prediction_info["abnormal_prob"],
                            "heart_rate": round(current_hr, 1),
                            "total_alerts": anomalies_detected,
                            "total_beats": total_beats,
                            "timestamp": time.time()
                        })

                if ws_server and sample_count % config.STREAM_BATCH_SIZE == 0:
                    payload = {
                        "type": "telemetry",
                        "raw_ecg": raw_ecg,
                        "filtered_ecg": round(filtered_ecg, 2),
                        "is_r_peak": is_r_peak,
                        "heart_rate": round(current_hr, 1),
                        "total_alerts": anomalies_detected,
                        "total_beats": total_beats,
                        "timestamp": time.time()
                    }
                    ws_server.broadcast(payload)

                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                now = time.monotonic()
                if last_pipeline_error_log is None or now - last_pipeline_error_log >= 5.0:
                    last_pipeline_error_log = now
                    print(f"[WARN] pipeline error (suppressed): {exc!r}", flush=True)
                if consecutive_errors >= max_consecutive_errors:
                    print(
                        f"[FATAL] {consecutive_errors} consecutive pipeline errors; aborting.",
                        flush=True
                    )
                    sys.exit(1)

    except KeyboardInterrupt:
        print("\nPipeline stopped.")
    finally:
        driver.close()
        elapsed = time.time() - start_time
        print(f"Elapsed: {elapsed:.1f}s | Beats: {total_beats} | Alerts: {anomalies_detected}")


if __name__ == "__main__":
    main()
