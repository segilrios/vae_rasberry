# Raspberry Realtime Bundle

Estructura incluida (copiar esta carpeta completa a la Raspberry):

- `VAE_implementation/scripts/07_pack_unpack.py`
- `VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py`
- `VAE_implementation/scripts/12_acq_sensor_to_zmq.py`
- `VAE_implementation/configs/vae_default.yaml`
- `VAE_implementation/models/GLOBAL_BEST/encoder_mu_int8.tflite`
- `data/processed/psd_1024/dataset_psd_1024_norm.npz`

## Instalacion en Raspberry

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pyzmq pyyaml tflite-runtime
```

Si no puedes instalar `tflite-runtime`, usa:

```bash
pip install tensorflow
```

## Ejecucion (2 procesos)

1) Adaptador adquisicion externa -> ZMQ IPC

```bash
python VAE_implementation/scripts/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "tu_modulo.tu_fuente:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

2) Bridge VAE -> UDP servidor

```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1 --log_every_packets 10
```

## Nota

Si la PSD de adquisicion ya llega normalizada [0,1], agrega `--already_normalized` al script `11_edge_hackrf_psd_zmq_to_udp.py`.
