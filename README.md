# DCNet, IEEE JSTARS 2024</a> </p>

- Paper: [Burned Area Segmentation in Optical Remote Sensing Images Driven by U-Shaped Multistage Masked Autoencoder](https://ieeexplore.ieee.org/document/10531632)


## Abstract

Computer vision (CV) for natural disaster monitoring from optical remote sensing images (ORSIs) has been an emerging topic in analyzing ORSIs. Recently masked autoencoder (MAE) has achieved great success in CV and shown promising potential for many downstream vision tasks. However, due to the inherent limitation of vision transformer (ViT) in MAE which has a fixed feature scale and performs poorly in modeling local spatial correlation, directly applying MAE to burned area segmentation (BAS) in ORSIs fails to achieve satisfactory results. To address this problem, we propose a novel dual-branch complement network (DCNet) driven by U-shaped multistage masked autoencoder (UMMAE) for BAS in ORSIs, which is also the first application of MAE in BAS. UMMAE has four stages and introduces skip connection between the encoder and decoder at the same stage, which improves the feature diversity and further enhances the model performance. DCNet has three major components: the ViT encoder (global branch), the convolution encoder (local branch), and the decoder. The global branch inherits visual representation learning ability from the pretrained UMMAE and captures global contextual information from the input image, while the local branch extracts local spatial information at different scales. Features from two different branches are fused in the decoder for feature complementation, which improves feature discriminability and segmentation accuracy. Besides, we build a new BAS dataset containing ORSIs of burned area in California, USA, from 2017 to 2022. Extensive experiments on two BAS datasets demonstrate that our DCNet outperforms the state-of-the-art methods.

## Dataset
- **CBAS**: CBAS dataset can be accessed at: https://pan.baidu.com/s/1QVuxCcQkuVU_eRH-FrcdoQ   code: su32
- **Bushfire**: <https://github.com/TangChao729/Burned_Area_Segmentation>

## Related Works
[Dual backbone interaction network for burned area segmentation in optical remote sensing images ](https://github.com/Voruarn/DBINet), IEEE GRSL 2024.

[Controllable diffusion generated dataset and hybrid CNN–Mamba network for burned area segmentation ](https://github.com/Voruarn/HCM), ADVEI 2025.

[A novel salient object detection network for burned area segmentation in high-resolution remote sensing images ](https://github.com/Voruarn/PANet), ENVSOFT 2025.

```
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

@article{10531632,
    author={Fu, Yuxiang and Fang, Wei and Sheng, Victor S.},
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
    title={Burned Area Segmentation in Optical Remote Sensing Images Driven by U-Shaped Multistage Masked Autoencoder}, 
    year={2024},
    volume={17},
    number={},
    pages={10770-10780},
  }
```
