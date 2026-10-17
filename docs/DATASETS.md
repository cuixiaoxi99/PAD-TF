# Dataset setup and provenance

PAD-TF does not redistribute third-party datasets. The bootstrap script downloads the public files from their original or official mirrors and stores them under `data/`.

## Sources

| Dataset | Values | Graph / metadata |
|---|---|---|
| PEMS03, PEMS04, PEMS07, PEMS08 | [FlashST-DATA](https://huggingface.co/datasets/bjdwh/FlashST-DATA) | FlashST-DATA |
| LargeST-CA (`CA_District5`) | [FlashST-DATA](https://huggingface.co/datasets/bjdwh/FlashST-DATA) | FlashST-DATA |
| METR-LA | [Zenodo record 5146275](https://zenodo.org/records/5146275) | [DCRNN sensor graph](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) |
| PEMS-BAY | [Zenodo record 5146275](https://zenodo.org/records/5146275) | [DCRNN sensor graph](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) |

P7/P8 use the 211-sensor `CA_District5` subset distributed for FlashST compatibility. They do not use the complete 8,600-sensor LargeST-CA release.

Please review and comply with each upstream dataset's license or terms before use.

## Expected layout

```text
data/
├── METR_LA/
│   ├── METR-LA.csv
│   └── adj_mx.pkl
├── PEMS_BAY/
│   ├── PEMS-BAY.csv
│   └── adj_mx_bay.pkl
├── PEMS03/
│   ├── PEMS03.npz
│   ├── PEMS03.csv
│   └── PEMS03.txt
├── PEMS04/
│   ├── PEMS04.npz
│   └── PEMS04.csv
├── PEMS07/
│   ├── PEMS07.npz
│   └── PEMS07.csv
├── PEMS08/
│   ├── PEMS08.npz
│   └── PEMS08.csv
└── CA_District5/
    ├── CA_District5.npz
    └── CA_District5.npy
```

## Automated download

From the repository root:

```bash
bash scripts/bootstrap_remote.sh
```

The script is restart-safe: already-present dataset files are not downloaded again. After setup, validate all paths and the pinned FlashST checkout with:

```bash
python -m padtf.cli doctor
```
