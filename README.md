#  On Loss Functions for Music Source Separation

This repository contains the PyTorch (1.2+) implementation of our branch of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch): a study on different loss functions for Music Source Separation.  For mor details on the architecture, visit the [main respository](https://github.com/sigsep/open-unmix-pytorch).

__Related Projects:__ open-unmix-pytorch | [musdb](https://github.com/sigsep/sigsep-mus-db) | [museval](https://github.com/sigsep/sigsep-mus-eval) | [norbert](https://github.com/sigsep/norbert)

## Loss coverage
Researchers have proposed several alternatives to L2 that can be found in recent literature, but usually tested in different conditions. These can be summarized in the following taxonomy:

![](https://lh3.googleusercontent.com/Qzz1J27PdnPRhriiTS7rvwUdmAIodurJz-2faS9jWkoHW6M6OBS0F1KCs9T5ilhmn-eQOz3wTpwy483DF8jJEorQdGDTr4df3PivCnwoAFSsdVu_F9gCKSgt8o71BXsgmX4CTvcwKNChYv-HWO3fNYbS08e5qq8q1_QCDCk4Xwk9dMAy4RFWCX6wrIfNSLDbh4x33YXP3RscTrsFWExaY_qdhZDZKEW-YXPgl03ZBNdUy54Ks_piXs8TevZN5u9KKWr6mejeh979ePEEQu9vU2y8QNp2Yf16aK0PbVvYpqxAqpnluwb3rQM7tVmq7QEu6oYh9SryeYmr8GewNHDJo-WsojT2dXclAd7pyZlzicSjkrmqAYKHzngoJgKEedudR6FdMmFafh3kMnucIXbVrsdyPBnitL87ONVBWhNyhlqhcQxBQ5qGmCzeEGAjfP4XTSc2NeqZrpQ-jiGNJKe4ocoaK1ThJX2hgnYUWHVf45ppTLUxNGcIHHodh8Ol0NT0toKMUQlGzP8H5bzGcLrt21SJgSO1NTAG9j4PZiQ2I38rLAv9wsvAOh7MS5ge_RC4-d2kwubCPbvIPWvH8CQyTD4SgeJVDIZHFxlCz-oB7TLcKzX18S6ugxFNgRy8mdYPa2pVcABlrj-6_Jh-aHtDbx4a1wEhrfTyj6Grf7JYTrLKPsRm3FJ7Rvkhwq5bqYLC4i61=w1920-h920-ft)

First and foremost, traditional spectrogram losses may be computed with an spectral source separator such as Open-Unmix. Next, performing the ISTFT inside the model additionally allows us to implement losses that are normally used in time domain models. Besides, we can adapt the whole training loop from supervised learning into an semi-supervised approach with adversarial losses. Lastly, deep features might be incorporated into the supervised loop through the use of pre-trained models' embeddings. This taxonomy categorization also depends on the point at which each loss is computed in the model's pipeline:

![](https://lh3.googleusercontent.com/zhbZHpaTdASaQQUlvEWF3A1MPC5rXzzH--3XnxYwSaVRJYmAVubBua66r4T6F11xXF1Hi45mRLdsDIYGv-rwVwx1S3a26l2FZDmvCnQI98pJJoIaJyVUxgMYM8qEH63dLzHCfCIuAmVrMypPsXZb72rKNDFH1wKMHyhmdBricvBwkiGrLVFN8xKA4ejksrnTDpZov2fOam05WK5s7i3iv1e3KPVBTTH-Pjb2EUX4pDiIrngLtNj-cVMt3whsFEyFJHNfx-8_956jDlqYbOFO7F4oHlP24xpwpjTPi4rAlJJ5kU8hk_W3kedfG3DlJrfb6Z-lVpQPyMFORmzo7K-23k97L4Hp4lxLQtmQTRCJG90TTNn85ZcPLdeEJC04ymAD4l-0uV8147tyicGg2_vC6E2TYyamW0JFdkGpMFYQoG5Tu_FbAO2NvNJxw_aGIhbEJjgZpzEd4umHprSNcszT1mbvdvP53IRMy2x-6o74f44nW74Xtx9hsbjDCMYIAZTtLc2_sFGEoqMfjKLC7_pKaEjCDX2vI95ba_Zn7fFe2hwBbDL3p3PqCxDRr0ZY0IJuwI1HNrWmgPoW05KfcETAtLO68jJsKPxZzOBIrTzNvSdkPugxy05GQ7Jr3MgBideOWYhq2UI46oAeFGbKxTtwI2CR3MWNfPbgMqvvfnQuUYIzTU-RiNnoMEZS_owU2HwUkiJM=w1440-h892-ft)

### Open-Unmix Adaptation

TODO

### Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _open-unmix_, simply run:
```
 conda install -c pytorch torchaudio 
```
```
pip install museval
```

`conda env create -f environment-X.yml` where `X` is either [`cpu-linux`, `gpu-linux-cuda10`, `cpu-osx`], depending on your system. For now, we haven't tested windows support.

### Replicating the experiments
Once installed, update the dataset and outpupt paths on `run_experiments.sh` and run it:
```bash
sh run_experiments.sh
```

### Reading results with pandas


### Results compared to SiSEC 2018 (SDR/Vocals)

Open-Unmix yields state-of-the-art results compared to participants from [SiSEC 2018](https://sisec18.unmix.app/#/methods). The performance of `UMXHQ` and `UMX` is almost identical since it was evaluated on compressed STEMS.

![boxplot_updated](https://user-images.githubusercontent.com/72940/63944652-3f624c80-ca72-11e9-8d33-bed701679fe6.png)

Note that

1. [`STL1`, `TAK2`, `TAK3`, `TAU1`, `UHL3`, `UMXHQ`] were omitted as they were _not_ trained on only _MUSDB18_.
2. [`HEL1`, `TAK1`, `UHL1`, `UHL2`] are not open-source.

#### Scores (Median of frames, Median of tracks)

|target|SDR  |SIR  | SAR | ISR | SDR | SIR | SAR | ISR |
|------|-----|-----|-----|-----|-----|-----|-----|-----|
|`model`|UMX  |UMX  |UMX  |UMX |UMXHQ|UMXHQ|UMXHQ|UMXHQ|
|vocals|6.32 |13.33| 6.52|11.93| 6.25|12.95| 6.50|12.70|
|bass  |5.23 |10.93| 6.34| 9.23| 5.07|10.35| 6.02| 9.71|
|drums |5.73 |11.12| 6.02|10.51| 6.04|11.65| 5.93|11.17|
|other |4.02 |6.59 | 4.74| 9.31| 4.28| 7.10| 4.62| 8.78|

## Training

Details on the training is provided in a separate document [here](docs/training.md).

## Extensions

Details on how _open-unmix_ can be extended or improved for future research on music separation is described in a separate document [here](docs/extensions.md).


## Design Choices

we favored simplicity over performance to promote clearness of the code. The rationale is to have __open-unmix__ serve as a __baseline__ for future research while performance still meets current state-of-the-art (See [Evaluation](#Evaluation)). The results are comparable/better to those of `UHL1`/`UHL2` which obtained the best performance over all systems trained on MUSDB18 in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app).
We designed the code to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. We favored framework specifics implementations instead of having a monolithic repository with common code for all frameworks.

## How to contribute

_open-unmix_ is a community focused project, we therefore encourage the community to submit bug-fixes and requests for technical support through [github issues](https://github.com/sigsep/open-unmix-pytorch/issues/new/choose). For more details of how to contribute, please follow our [`CONTRIBUTING.md`](CONTRIBUTING.md). For help and support, please use the gitter chat or the google groups forums. 

### Authors

[Fabian-Robert Stöter](https://www.faroit.com/), [Antoine Liutkus](https://github.com/aliutkus), Inria and LIRMM, Montpellier, France

## References

<details><summary>If you use open-unmix for your research – Cite Open-Unmix</summary>
  
```latex
@article{stoter19,  
  author={F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji},  
  title={Open-Unmix - A Reference Implementation for Music Source Separation},  
  journal={Journal of Open Source Software},  
  year=2019,
  doi = {10.21105/joss.01667},
  url = {https://doi.org/10.21105/joss.01667}
}
```

</p>
</details>

<details><summary>If you use the MUSDB dataset for your research - Cite the MUSDB18 Dataset</summary>
<p>

```latex
@misc{MUSDB18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}
```

</p>
</details>


<details><summary>If compare your results with SiSEC 2018 Participants - Cite the SiSEC 2018 LVA/ICA Paper</summary>
<p>

```latex
@inproceedings{SiSEC18,
  author="St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Ito, Nobutaka",
  title="The 2018 Signal Separation Evaluation Campaign",
  booktitle="Latent Variable Analysis and Signal Separation:
  14th International Conference, LVA/ICA 2018, Surrey, UK",
  year="2018",
  pages="293--305"
}
```

</p>
</details>

⚠️ Please note that the official acronym for _open-unmix_ is **UMX**.

### License

MIT

### Acknowledgements

<p align="center">
  <img src="https://raw.githubusercontent.com/sigsep/website/master/content/open-unmix/logo_INRIA.svg?sanitize=true" width="200" title="inria">
  <img src="https://raw.githubusercontent.com/sigsep/website/master/content/open-unmix/anr.jpg" width="100" alt="anr">
</p>
