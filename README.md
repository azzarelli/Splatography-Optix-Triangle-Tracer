# Extending Splatography for VFX using Optix-Triangle-Tracer

Developing an environment to relight dynamic Gaussian Splatting scenes using synthetic lighting via Optix-raycasting libraries.

Underlying pipeline:
1. Used the [ViVo dataset](https://vivo-bvicr.github.io/) to train a sparse-view scene using the [Splatography](https://azzarelli.github.io/splatographypage/index.html) dynamic GS model for filmmaking applications
2. Implemented a ray-tracing interface that estimates first-hit indices of Gaussians 