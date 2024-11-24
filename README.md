<h1>
Understanding Multimodal Large Language Models (MLLMs)
</h1>

<h2>
Introduction
</h2>

<p>
In 2017, the groundbreaking paper “Attention Is All You Need” introduced the transformer architecture, which revolutionized the machine learning landscape. By placing attention mechanisms at the core of deep learning models, transformers redefined how machines process sequential data, enabling them to focus selectively on relevant information. This innovation took the ML community by storm, giving rise to a new era of state-of-the-art models across various domains.

The applications of attention have since permeated every corner of artificial intelligence. Large Language Models (LLMs), such as ChatGPT, have leveraged transformers to redefine natural language understanding and generation, powering applications like conversational agents, summarization tools, and code generation. Similarly, Vision Transformers (ViTs) adapted the transformer architecture to computer vision, enabling breakthroughs in image classification, object detection, and beyond. Attention’s ability to handle long-range dependencies and multimodal inputs has made it the cornerstone of modern AI advancements.

Recent progress has further pushed the boundaries of what attention mechanisms can achieve. The development of Multimodal Large Language Models (MLLMs) marks the next step in this journey. These models integrate multiple data modalities—like text, images, and audio—into a unified framework, enabling them to tackle tasks that were previously impossible. From Flamingo to PaliGemma, MLLMs vary widely in their architecture, parameter counts, and performance, ranging from lightweight models suitable for edge devices to massive systems trained on billions of examples for unparalleled generalization.

In this blog, we’ll focus on PaliGemma (3B), a cutting-edge MLLM known for its ability to combine high-dimensional features from images and text using separate image and text encoders, using attention mechanisms to generate a common feature space. By exploring its architecture and functionality, we’ll uncover how MLLMs are shaping the future of AI.
</p>

<h2>
Image and Text Encoders
</h2>

The process of transforming raw data into meaningful representations lies at the heart of multimodal learning, and this task is performed by encoders. Encoders play a crucial role in extracting features from raw inputs, such as images or text, and preparing them for downstream tasks. In the context of Multimodal Large Language Models (MLLMs), this step ensures that information from different modalities can be aligned and integrated effectively in a shared feature space.

There are two primary approaches to designing image and text encoders:

1. Traditional Approach: Building Custom Pipelines
In this approach, specialized pipelines are constructed for each modality to extract features. For images, this typically involves Vision Transformers (ViTs) or convolutional neural networks to process pixel data and generate feature embeddings. For text, tokenizers convert sentences into discrete tokens, which are then passed through transformer layers to produce continuous feature representations.

The extracted features are then projected into a common feature space, enabling alignment and comparison between modalities. A contrastive loss setting is commonly used during training to ensure that semantically similar image-text pairs are closer in the feature space, while dissimilar pairs are pushed apart. This method has been successfully employed in early multimodal models like CLIP and ALIGN, which demonstrated remarkable performance on tasks like zero-shot image classification and cross-modal retrieval.

2. Modern Approach: Leveraging Pretrained Models
The second, more recent approach builds on the idea of transfer learning by utilizing pretrained encoders for image and text features. Instead of designing modality-specific pipelines from scratch, this method uses models like ViTs pretrained on large-scale datasets for image feature extraction and LLMs such as GPT for text. The extracted features are mapped to a shared feature space through learnable transformations, often using linear layers or specialized fusion mechanisms.

The advantage of this approach lies in its flexibility and scalability. By decoupling feature extraction from multimodal alignment, architectures can focus on learning to integrate information across modalities. Notable examples include LLaMA, PaliGemma, and GPT models, which excel in aligning high-dimensional features from image and text encoders through advanced attention mechanisms.

**Why Modern Approaches Dominate**
There are two primary reasons : Firstly, it is difficult, computationally heavy and requires additional parameters to bring images and text with more context and details to a common feature space. The data used to train CLIP was relatively simpler, with sentences like : A cat playing with a ball. Internally, the model learns to map the words "cat" and "ball" to the cat and ball objects in the image. This is not easily achievable when there in more detail and context in the inputs. This is the reason why CLIP is very useful in tasks like classification, and cannot be used for autoregressive text generation.
Secondly, transfer learning has proven to be really efficient, and requries less compute and fewer parameters to achieve similar results. 
The shift toward pretrained models reflects a growing emphasis on leveraging existing architectures with proven capabilities. Models like PaliGemma exemplify this paradigm, using powerful pretrained encoders to extract features and concentrating the training effort on developing a deeper cross-modal understanding. This strategy not only reduces computational costs but also enables the model to achieve state-of-the-art performance on multimodal tasks.
By combining the strengths of pretrained encoders with attention-based fusion techniques, MLLMs have redefined how information from diverse modalities can be brought together in a seamless and efficient manner. In the next section, we will dive deeper into the mechanism that makes this integration possible: attention.
Attention is all you need : 
Attention is the cornerstone of modern AI architectures, enabling models to focus selectively on relevant parts of input data while processing sequences. Introduced in the “Attention Is All You Need” paper, this mechanism has transformed natural language processing, computer vision, and multimodal learning by providing a robust framework for capturing dependencies and relationships across elements in data. The complete attention mechanism in all its integrity is beyond the scope of this blog, but here's a quick overview. 

At its core, the attention mechanism revolves around three key components: Query (Q), Key (K), and Value (V) matrices. These matrices are derived from the input data, where:

Query (Q): Represents the item for which the model seeks relevant context.
Key (K): Represents the items against which relevance is measured.
Value (V): Contains the information to be aggregated based on the relevance scores.
The relevance of a query to a key is computed as a scaled dot product:
			Attention (Q, K, V)​ = softmax(QKT/sqrt(dk)).V
where dK  is the dimensionality of the key vectors, and the softmax function ensures that the attention scores are normalized, effectively weighing the contributions of different values.

To preserve sequence information, sinusoidal positional embeddings are added to the input representations before processing. These embeddings encode relative positions of tokens or image patches in the sequence, allowing the model to differentiate between, for instance, "dog bites man" and "man bites dog." The embeddings are designed as a series of sine and cosine functions that capture positional patterns in a continuous space.

Types of Attention Mechanisms
Different tasks and architectures call for varying forms of attention. The most common types include:

Self-Attention: Computes attention scores within the same input sequence, enabling a token to attend to other tokens in its context. This is the backbone of models like transformers.
Cross-Attention: Computes attention scores between two sequences, such as between text and image features in multimodal models.
Global vs. Local Attention: Global attention considers all tokens, while local attention focuses on a subset, optimizing for tasks with very long sequences.
I am not describing the concept of multiheaded attention here as it is beyond the scope of this book, but it is to be noted that we'll be averaging over all the attention heads in each layer for the upcoming discussions. 

**Overview of the Complete PaliGemma model Architecture**
The PaliGemma model is designed to seamlessly integrate image and text data using a sophisticated architecture that leverages pretrained encoders and attention mechanisms. This section provides a detailed breakdown of its components and functionality.

Image Encoder: SiGLiP-400M
The image encoder used in PaliGemma is a modified version of the SiGLiP-400M model. It consists solely of the vision tower from the original SiGLiP architecture, comprising 26 layers of multi-headed self-attention networks. SiGLiP is trained on image-text pairs using a CLIP-like framework but replaces the softmax function with a sigmoid activation, which eliminates batch size dependency and accelerates training.

When an image is passed through the encoder, it is first divided into 256 patches, each of which is projected into a 1152-dimensional feature space, resulting in an output of dimensions [256, 1152]. These features are then passed through a linear layer to transform them into a 2048-dimensional feature space, matching the text encoder’s output dimensions.

Text Encoder: Gemma 2.0
The text encoder, Gemma 2.0, is a pretrained model specifically designed to tokenize sentences and generate feature embeddings. It consists of 17 layers of attention networks, followed by multi-layer perceptron (MLP) blocks for feature refinement.

The text input, once tokenized, is projected into a 2048-dimensional embedding space. The output dimensions are [number of tokens + 2, 2048], where the extra two tokens correspond to:

<bos>: The beginning-of-sentence token.
\n: The newline character, which aids in sequence demarcation.
Combining Features
Once the image and text features are processed, the outputs are concatenated to create a unified input for the Gemma decoder, which generates outputs in an autoregressive manner. The Gemma decoder is responsible for modeling the relationships between the image and text features, leveraging attention mechanisms to create a coherent understanding of multimodal data.

Image Features: 
[256,1152] → Projected via a linear layer → [256, 2048]
Text Features: 
[numberoftokens+2,2048]

Concatenation: The two transformed feature sets are concatenated along the sequence dimension and fed into the decoder.

An important detail is that the spatial information of image patches is lost when the features are passed through the linear layer. However, this information is not discarded internally—its just that it cannot be visualized as expected.
Training the Model
The PaliGemma model is trained end-to-end, with no components frozen during training. This ensures that both the image encoder and text encoder can adaptively refine their representations based on the multimodal input. The training process aligns the representations of text and image modalities, creating a shared feature space that enables cross-modal tasks.

PaliGemma’s architecture highlights a thoughtful integration of pretrained encoders and attention mechanisms, balancing efficiency with performance. In the next sections, we will explore how attention mechanisms are applied across the combined features and visualize their effects on model outputs.

**Plotting Techniques for the Image Encoder**
The SiGLiP image encoder in PaliGemma comprises 26 layers of multi-headed self-attention networks. Each layer attends to different aspects of the image, focusing on specific regions and patterns. By plotting the attention maps, we can gain insight into how the model interprets visual data and what features it prioritizes during processing.

For a thorough understanding, we will visualize two types of attention maps:

Layer-Specific Attention: Attention map from the first layer, representing low-level feature attention.
Global Attention: A cumulative attention map obtained by summing the attention scores across all 26 layers, illustrating the broader context captured by the model.
Steps to Plot an Attention Map
The process of generating an attention map involves extracting and processing the attention scores from the encoder’s attention layers:

Extracting Attention Scores:
The attention scores matrix  
			attention_scores = A.KT
for each attention layer has a shape of [256, 256], corresponding to the 256 image patches. Each value in this matrix describes how much one patch attends to another.

Reshaping to Preserve Spatial Layout:
To visualize attention in the spatial domain, we reshape the attention scores from [256, 256] to [16, 16, 16, 16].

The first two dimensions [16,16] represent the spatial grid of patches in the original image.
The last two dimensions  [16,16] describe the attention relationships between patches.

Reducing Dimensionality:
To aggregate the attention information, we compute the maximum attention scores across the last two dimensions, reducing the matrix to [16, 16]. This step creates a simplified map of how each patch attends to all others in the spatial grid.

Upscaling to Image Resolution:
The [16, 16] attention map is upscaled to the original image dimensions using bilinear interpolation. This ensures the map aligns with the input image resolution for effective visualization.

Overlaying on the Original Image:
The resulting attention map is superimposed on the original image with a transparency value. This overlay helps visualize which parts of the image the encoder attends to, either at a single layer or across multiple layers.

Results from Image Encoder Visualizations
Visualizing attention maps from the image encoder provides valuable insights into how the SiGLiP model processes visual data at different levels. Here are the key observations:

Single-Layer Attention Maps:

These maps demonstrate a localized focus, highlighting specific regions or features within the image.
The model tends to concentrate on distinct patches, such as edges, textures, or parts of objects, without forming a holistic understanding of the entire image.
This behavior aligns with the lower-level processing characteristic of initial attention layers, where the focus is primarily on extracting granular details.
Global Attention Map (Summation of All Layers):

The aggregated map illustrates how the model combines information from all 26 layers to develop a comprehensive understanding of the image.
Specific regions, typically corresponding to objects or areas of interest, receive higher attention, while background regions are assigned comparatively lower attention.
This distribution reflects the model’s ability to prioritize meaningful features while suppressing irrelevant details, a critical factor in effective feature extraction.

**Plotting Techniques for a Combination of Image and Text Encoders**

Visualizing the attention between image and text encoders is a more complex task compared to analyzing image-only attention maps. This complexity arises from the loss of spatial information in the image features due to a linear layer transformation and the distribution mismatch between image and text features. However, with innovative techniques, we can gain meaningful insights into how words interact with the image at a high level.

Challenges in Visualization
Loss of Spatial Information:
The image features are flattened and passed through a linear layer, causing the spatial relationships among image patches to be lost. This makes direct patch-based attention visualization infeasible.

Feature Distribution Differences:
Image and text features originate from different encoders, each operating in distinct feature spaces with unique means and variances. This discrepancy prevents straightforward comparisons of attention scores in a shared attention map.

Approach to Understand Image-Text Attention
Instead of attempting to directly map specific words to specific patches in the image, we analyze the total attention that each word assigns to the entire image. This approach allows us to derive meaningful insights without needing direct spatial correspondence.

**Steps for Visualization:**

Extract Attention Matrices:
From the Gemma decoder, extract the attention matrix with dimensions [256 + num_words + 2, 256 + num_words + 2]. This matrix contains attention scores for both the image patches and text tokens, organized as follows:
[:256,:256]: Image-Image Attention Matrix.
[:256,256:]: Image-Text Attention Matrix.
[256:,:256]: Text-Image Attention Matrix.
[256:,256:]: Text-Text Attention Matrix.
Focus on the Image-Text Attention Matrix:
Select the matrix 
[:256,256:], which has the shape [256, num_words + 2]. This matrix describes how each image patch attends to each word (including the  <bos> and  tokens).

Reshape for Spatial Understanding:
Reshape the matrix into a shape of [16, 16, num_words + 2]. This reorganization treats the 256 image patches as a 16x16 grid, restoring a spatial interpretation.

Visualize Attention for Each Word:
For each word (i.e., for each value in the last dimension of the reshaped matrix), generate a 16x16 attention map. These maps quantify how much attention the word gives to the entire image.

Overlay Attention Maps on the Image:
Each attention map is plotted over the original image, allowing a comparative study of how different words interact with the visual input.

Key Results from Visualization
This method yields num_words + 2 attention maps, one for each word token. By analyzing the variation in these maps for different prompts, we can study how the model dynamically distributes attention between the textual and visual modalities. These insights will guide us toward meaningful conclusions about the attention mechanism in multimodal settings, as detailed in the next section.

**Drawing Results from Image-Text Attention Maps**
While a single attention map for an image-text pair might seem uninformative, meaningful patterns and insights emerge when comparing maps across different input combinations. By systematically varying the image-text pairs and studying the resulting attention distributions, we can uncover how the model aligns the visual and textual modalities.

Experimental Setup
To explore the behavior of the attention mechanism, we will use two images (one of a cat and one of a dog) and six text prompts with subtle variations. The prompts are as follows:

"Is this an image of a cat or a dog?"
"Is this an image of a dog or a cat?"
"Is this a cat?"
"Is this a dog?"
"This is a dog."
"This is a cat."
Analysis Framework
The analysis will focus on comparing the image-text attention maps generated by the second attention matrix  [:256,256:], which quantifies how each word attends to the entire image.

**Conclusions from Plotted Attention Maps**

The analysis of the attention maps from image-text input pairs reveals several notable patterns and insights about how the PaliGemma model aligns textual and visual modalities. Below are the primary conclusions drawn:

1. Consistent Relative Attention Scores Across Prompts
Upon comparing attention maps for various prompts across the two images (cat and dog), a clear pattern of similarity emerges for words other than "cat" and "dog."

These words (e.g., "this," "is," or "a") exhibit attention maps that consistently highlight the same regions across all prompts.
This consistency suggests that the model treats these auxiliary words uniformly, focusing on certain background regions of the image regardless of the main subject.
2. Impact of Keywords on Attention Maps
For the prompts "This is a cat" and "This is a dog," the attention maps reveal significant changes only for the keywords ("cat" and "dog").

The attention distribution for all other words remains unchanged, indicating that these keywords play a critical role in the model’s decision-making process.
This demonstrates that the model identifies and prioritizes the semantically meaningful tokens (in this case, the words describing the objects in the image) over auxiliary tokens.
3. Object Localization and Semantic Understanding
The attention maps show a notable concentration of attention for the words "cat" and "dog" compared to other words in the prompt.

This implies that the model has effectively learned to localize objects in the image based on their textual descriptions.
The high attention scores for these keywords confirm that the model associates specific visual regions of the image with their corresponding textual labels, highlighting its ability to process and align multimodal inputs.
Summary
The plotted attention maps provide valuable insights into the inner workings of the attention mechanism in multimodal models like PaliGemma. These findings demonstrate the model's ability to:

Maintain consistent processing for non-essential words.
Assign higher importance to semantically critical words like "cat" and "dog."
Localize objects in images based on textual prompts, showing an advanced level of cross-modal understanding.
These observations validate the efficacy of the model’s design while also offering a foundation for future improvements in multimodal attention-based architectures.

**Applications and Conclusion**

Applications of Understanding Internal Model Functioning
Explainability in Large Models

The ability to analyze what a model learns internally is a significant step toward addressing the explainability problem in LLMs and MLLMs.
These models are often regarded as black boxes due to their massive scale—both in terms of training data and the number of parameters. By visualizing and interpreting attention maps, we can shed light on how models process inputs and make decisions.
Enhancing Model Quality

Gaining insights into the attention mechanisms and feature alignments can lead to better-informed architectural decisions.
This understanding enables researchers to identify weaknesses in the current design and develop more efficient and accurate models for future applications.
Improved Generalization Across Modalities

Studying the internal workings of these models helps uncover patterns that can enhance their generalization abilities.
With a better grasp of how models handle diverse inputs, we can design training protocols that make them more robust across different domains and input types.
Conclusion
In this blog, I explored the inner workings of the PaliGemma multimodal model by visualizing its attention mechanisms and feature interactions. Through attention maps, we observed how the model aligns image and text features, focusing on meaningful regions and keywords while maintaining consistency in auxiliary information processing.

This exercise not only highlights the potential of attention-based architectures but also demonstrates the importance of understanding the learning process of large models. By visualizing these complex processes, I hope to make the topic more accessible, stimulate curiosity, and inspire further exploration into the world of explainable and generalizable machine learning models.

This journey into attention maps is just the beginning—there’s much more to uncover in the pursuit of explainability and improvement of AI systems.