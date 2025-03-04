# <img src="assets/logo-Photoroom.png" style="vertical-align: -10px;" :height="60px" width="60px"> *Retrieval-Augmented Perception*

This repo contains the official code for the paper ["***Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG***"](https://arxiv.org/abs/2503.01222).

## 💡 Highlights

- 🔥 We propose ***RAP***, a training-free framework designed to enhance Multimodal Large Language Models' (MLLMs) ability to process high-resolution images effectively.

## 📜 News
**[2025.03.04]**  🚀 We released the [ArXiv paper](https://arxiv.org/abs/2503.01222).

## 👀 Introduction

High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To overcome the limitations of existing methods, this paper shifts away from prior dedicated heuristic approaches and revisits the most fundamental idea to HR perception by enhancing the long-context capability of MLLMs, driven by recent advances in long-context techniques like retrieval-augmented generation (RAG) for general LLMs.  Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose ***Retrieval-Augmented Perception (RAP)***, a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed *Spatial-Awareness Layout*. To accommodate different tasks, the proposed *Retrieved-Exploration Search (RE-Search)* dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of ***RAP***, with LLaVA-v1.5-13B achieving a 43% improvement on $V^*$ Bench and 19% on ***HR-Bench***.
![](assets/motivation.png)


## 🧑🏻‍💻 Code

*Coming Soon.*



## 📧 Contact
- Wenbin Wang: wangwenbin97@whu.edu.cn 

## ✒️ Citation
If you use *RAP* in your research, please cite our work:
```
@article{rap,
      title={Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG}, 
      author={Wenbin Wang and Yongcheng Jing and Liang Ding and Yingjie Wang and Li Shen and Yong Luo and Bo Du and Dacheng Tao},
      year={2025},
      journal={arXiv preprint},
      url={https://arxiv.org/abs/2503.01222}, 
}
```


## Acknowledgement
This work is built upon the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
