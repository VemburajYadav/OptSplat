<p align="center">
  <h1 align="center">OptSplat: Recurrent Optimization for Generalizable Reconstruction and
Novel View Renderings</h1>
  <p align="center">
    <a href="https://github.com/VemburajYadav/">Vemburaj Yadav</a>
    &nbsp;·&nbsp;
    <a href="https://av.dfki.de/members/pagani/">Alain Pagani</a>
    &nbsp;·&nbsp;
    <a href="https://av.dfki.de/members/stricker/">Didier Stricker</a>

  </p>
  <h3 align="center">Under Review </h3>
  <h3 align="center"><a href="https://www.dropbox.com/scl/fi/wyv27h9uinfivfh4gxzu7/OptSplat.pdf?rlkey=cqc3zd55jyc9laz4y45zfd91g&st=vw71fezc&dl=0">Paper</a> | <a href="https://VemburajYadav.github.io/OptSplat/">Project Page</a> | <a href="https://drive.google.com/drive/folders/1cqZx2Vl8Vf6XtpPPG2QA52GmjjDa3AkL?usp=drive_link">Pretrained Models</a> </h3>

<p align="center">
  <a href="">
    <img src="https://VemburajYadav.github.io/OptSplat/static/images/IterPredictions2.png" alt="Logo" width="100%">
  </a>
</p>

<p align="center">
<strong>OptSplat encompasses an recurrent optimization framework for the iterative refinement and joint estimation of input view depth and novel view synthesis.</strong> <br>
</p>

<br>
</p>


## Installation

To get started, clone this project, create a conda virtual environment using Python 3.10+, and install the requirements:

```bash
git clone https://github.com/VemburajYadav/OptSplat.git
cd OptSplat
conda create -n optsplat python=3.10
conda activate optsplat
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Compile the CUDA kernels to use the memory-efficient implementation for cost volume look-ups (applicable only during inference) 

```bash
pip install .
```
## Acknowledgements

This project is developed with several fantastic repos: [MVSplat](https://github.com/donydchen/mvsplat), [DepthSplat](https://github.com/cvg/depthsplat), [RAFT](https://github.com/princeton-vl/RAFT), [DPVO](https://github.com/princeton-vl/DPVO), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank the original authors for open-sourcing their excellent work.
