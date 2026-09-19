# Paper order for ICLR 2027 submission:
We will follow the same order of information flow as we did in the abstract because the abstract is like a small paper.

## Abstract
Hyperspectral imaging (HSI) captures dense reflectance spectra across hundreds of contiguous spectral bands. Robust representation learning of this data is central to applications like remote sensing, mineral mapping, atmospheric characterization etc. However sensor degradation, transmission failures, and external noise routinely corrupt these cubes, motivating a need for variational representation learning, typically via VAEs, to compress and probabilistically reconstruct corrupted spectra. Conventional VAEs apply convolutions that mix spatial and spectral information indiscriminately, blurring spectral signatures and causing posterior collapse or physically improbable hallucinations. Thus, we introduce PRISM (Physics-Informed Representation for Isolated Spectral-Spatial Modeling). PRISM is a dual-stream VAE routing spatial and spectral information through two independently-supervised spatial and spectral branches that re-couple only under a Spectral Angle Mapper (SAM) bounded fusion. This lets PRISM retain the spatial fidelity of standard 2D VAEs without sacrificing chemical accuracy; gaining 6.6 dB PSNR and 4.5 degree SAM improvement over rival architectures across the ablation grid. This decoupled latent is designed as a backbone for Latent Diffusion Models (LDMs) for purifying deep-space transmission corruptions and generating omitted frames. It already supports chemical interpolation for mineral discovery while offering a scalable framework for other physics-heavy domains like precision agriculture and climatology.

## Introduction:
1. What are Hyperspectral Images?
2. Where are Hyperspectral Images used?
3. Use of representation learning in the hyperspectral image domain for image compression, embedding creation and latent input creation for modeling frameworks like LDMs.
4. Variational nature of representation learning is motivated by the objective to fix/plausibly regenerate noisy and/or corrupted parts in an HSI.
5. VAEs offer a great solution for this via their ability to model smooth latent spaces that allow interpolation of chemical signature when performing denoising, new chemical generation spectra etc.
6. The issues suffered by standard VAEs (2D CNN variant only)
7. Introduce PRISM here with a short description of the architecture proposed. (Describe in 2 lines max because this will be the center of attraction in section 4 methodology)
8. List out the main contributions of the paper.
    1. Introducing a split stream architecture for self supervised representation learning.

## Related works:
Bucketize this section into related works based on type of work:
1. Work done in representation learning and spectral unmixing for HSI. (in general and not focused on VAEs)
2. Work done in optimising training using a penalty of differentiable physics term.
3. Work done in using specialized adaptation of VAE architecture for representation learning of HSI. (make sure to include the paper for vae-1d-pixelwise and its work on trying different objectives, vae-3d-spatio-spectra and other papers)

## Methodology:
1. PRISM Architecture
    1. In detail describe the architecture proposed.
    2. How the model splits into 2 individually supervised streams.
    3. Write proper description of the spatial stream, spectral stream and the shared reparameterization
    4. This section will also have 2 diagrams (that I’ll add later). The first one will be an architecture diagram (level 1 DFD). The second one will be a level 2 DFD showing the movement of the tensor through the two layers.
2. Objective Function and Physics Informed Fusion
    1. Describe the addition of the physics penalty term over the existing ELBO loss.
    2. What the net loss function is.
    3. What is the weight of the beta-term and the weight of SAM loss is.
3. Experiment Setup
    1. Describe the ablation setup. We are comparing our model (vae-our-nl) against 3 models i.e. standard vae, 1d pixelwise vae and 3d-spatio-spectra-vae.
    2. Write in paper that these architectures were initialised as described in their papers.
    3. We are using 3 datasets. IIRS, CRIMS and AVIRIS.
    4. Describe how the packing logic is. Write this in a simple language like “We split each cube into smaller 64x64xC cubes and then create a combined packed set of fixed number of patches to train on.
    5. For the datasets, use a central table describing the source of data, number of bands, wavelengths it covers, number of patches in training set, number of patches in validation set and number of patches in test set etc.

## Results and Discussions
1. Before we begin with any actual results, we need to show that the architecture passes the probes we had originally designed to check if the model learns something meaningful and informative over a mean reconstruction latent or random reconstructions.
2. This section is mainly based on 4 tables.
    1. The first table shows the reconstruction quality of the models.
    2. The second table shows the recovery from injected noise.
    3. The third table shows how smoothly we can interpolate chemical signatures
    4. The fourth table shows missing-pixel recovery.
3. Under each table, make sure to add a natural language read of what the table is telling us about the performance of the architecture.
4. End this section with a short paragraph giving a combined read.
## Conclusion
## AI Use Statement:
We have used AI to “Yes, to aid or polish writing. Details are described in the paper.” and “Yes, to draft sections of the paper. Details are described in the paper.” (We have ticked these options in the openreview submission page)

## Ethics Statement

## Reproducibility Statement

## References