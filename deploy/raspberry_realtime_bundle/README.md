# Raspberry Realtime Bundle - Paso a Paso (HackRF + VAE)

Este bundle está listo para operar en Raspberry con **normalización per-frame minmax** y reconstrucción en PC usando min/max por frame.

## 1) Verifica que el TFLite ya esté copiado
Ruta esperada:

```
deploy/raspberry_realtime_bundle/VAE_implementation/models/runs/run_fm_ext_001_peak_v4/encoder_mu_int8.tflite
```

## 2) Arranca el motor RF (HackRF)
En la Raspberry (terminal A), dentro del repo SDR:

```
export IPC_ADDR="ipc:///tmp/rf_engine"
./rf_app
```

Deberías ver:
```
[RF] Starting Engine. IPC=ipc:///tmp/rf_engine
```

## 3) Enviar control + reenviar PSD al IPC de VAE
En otra terminal (B), dentro del bundle:

```
python VAE_implementation/scripts/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --out_key psd_dbm \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":0,"vga_gain":40,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0 \
  --log_every 1
```

## 4) Edge VAE -> UDP (per-frame minmax)
En otra terminal (C), dentro del bundle:

```
python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --run_name run_fm_ext_001_peak_v4 \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --psd_key psd_dbm \
  --dest_ip <IP_PC> --port 5005 \
  --block_len 30 --zlib_level 1 \
  --packet_interval_ms 1000 \
  --log_every_packets 10
```

Notas:
- `normalize_mode` ya es `per_frame_minmax` en el YAML del bundle.
- Los min/max por frame se incluyen automáticamente en el paquete.

## 5) Receiver en PC (reconstrucción a escala original)
En el PC:

```
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --run_name run_fm_ext_001_peak_v4 \
  --bind_ip 0.0.0.0 --port 5005 \
  --save_every_packets 10 \
  --invert_norm_to_original
```

### Salida esperada
- `VAE_implementation/models/runs/run_fm_ext_001_peak_v4/udp_prod/`
  - `waterfall_recon.png`
  - `psd_last.png`
  - `udp_prod_metrics.json`

## 6) Métricas incluidas
Los logs de `EDGE11` y `SEND10` ahora muestran:
- `frames/s`
- `KB/s`
- `avgB/f`
- `ramMB` (si `psutil` está instalado)

---

### Instalación de psutil (opcional, recomendado)
En la Raspberry:
```
pip install psutil
```

---

Si necesitas fijar un flujo aún más constante, ajusta `--packet_interval_ms`.
