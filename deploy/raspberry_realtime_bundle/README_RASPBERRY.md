# Raspberry Realtime Bundle

Bundle para:
- streaming VAE en tiempo real
- captura de dataset extendido desde `rf_engine`
- guardado directo de `capture.csv` en una carpeta compartida del PC

## Archivos incluidos

- `VAE_implementation/scripts/07_pack_unpack.py`
- `VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py`
- `VAE_implementation/scripts/12_acq_sensor_to_zmq.py`
- `VAE_implementation/scripts/13_rf_engine_controller_to_zmq.py`
- `VAE_implementation/scripts/14_generate_acquisition_plan.py`
- `VAE_implementation/scripts/15_build_extended_dataset_index.py`
- `VAE_implementation/scripts/16_run_acquisition_session.py`
- `VAE_implementation/configs/vae_default.yaml`
- `VAE_implementation/configs/acquisition_experiment_extended.yaml`
- `VAE_implementation/models/GLOBAL_BEST/encoder_mu_int8.tflite`
- `data/processed/psd_1024/dataset_psd_1024_norm.npz`

## Instalacion en Raspberry

```bash
python3 -m venv .venv_edge
source .venv_edge/bin/activate
pip install --upgrade pip
pip install numpy pyzmq pyyaml
```

Si `tflite-runtime` existe para tu plataforma:

```bash
pip install tflite-runtime
```

Si no existe, usa `tensorflow` solo para el flujo de streaming, no para captura:

```bash
pip install tensorflow
```

## Guardar los CSV directamente en el PC

La forma correcta es compartir una carpeta de Windows por SMB y montarla en Raspberry.

### 1. En Windows

Comparte una carpeta, por ejemplo:

`C:\Users\gilse\OneDrive\Escritorio\VAE\kl_psd\data\raw\extended_acquisition`

Usa un nombre de recurso compartido simple, por ejemplo:

`extended_acquisition`

### 2. En Raspberry

Crea punto de montaje:

```bash
sudo mkdir -p /mnt/vae_pc_data
```

Monta la carpeta compartida:

```bash
sudo mount -t cifs //192.168.0.112/extended_acquisition /mnt/vae_pc_data -o username=<USUARIO_WINDOWS>,password=<PASSWORD_WINDOWS>,uid=$(id -u),gid=$(id -g)
```

Verifica:

```bash
ls /mnt/vae_pc_data
```

## Generar el plan

Esto normalmente se hace en el PC, pero si quieres tambien puede correrse en Raspberry.

```bash
python VAE_implementation/scripts/14_generate_acquisition_plan.py \
  --config VAE_implementation/configs/acquisition_experiment_extended.yaml \
  --max_sessions 1
```

## Ejecutar una sesion del plan

### 1. Arranca `rf_app`

```bash
cd ~/SDR-SpectrumMonitoring-Sensor
source venv/bin/activate
export IPC_ADDR="ipc:///tmp/rf_engine"
./rf_app
```

### 2. Ejecuta una fila del plan y guarda el CSV en la carpeta montada del PC

Desde el bundle:

```bash
cd ~/vae_rasberry/deploy/raspberry_realtime_bundle
source .venv_edge/bin/activate
python VAE_implementation/scripts/16_run_acquisition_session.py \
  --plan_csv /mnt/vae_pc_data/fm_extended_v1/acquisition_plan.csv \
  --base_capture_dir /mnt/vae_pc_data \
  --row_index 0
```

Esto genera algo como:

`/mnt/vae_pc_data/fm_extended_v1/000001_bogota_norte_urban_dense_r1/capture.csv`

Como `/mnt/vae_pc_data` apunta al PC, el archivo queda guardado directamente en Windows.

## Streaming VAE al servidor

Si ademas quieres enviar al PC por UDP:

### Terminal A

```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip 192.168.0.112 --port 5005 \
  --block_len 30 --zlib_level 1 --log_every_packets 1
```

### Terminal B

```bash
python VAE_implementation/scripts/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --out_key psd_dbm \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":0,"vga_gain":40,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0 \
  --log_every 1
```

## Consolidar dataset en el PC

Una vez tengas muchas sesiones:

```powershell
python VAE_implementation/scripts/15_build_extended_dataset_index.py --input_root data/raw/extended_acquisition/fm_extended_v1 --output_dir data/raw/extended_acquisition/fm_extended_v1_index
```
