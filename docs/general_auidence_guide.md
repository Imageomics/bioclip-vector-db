```
Document Type: User Guide

Target Audience: Non-technical users (e.g., citizen scientists, biology students, nature enthusiasts) who have no background in artificial intelligence or high-performance computing.

Primary Goals:
	- Demystify Technology: Explain AI concepts (embedding, indexing) using accessible analogies like "digital fingerprints" and "library sections."
	- Enable Usability: Define the search control sliders (Search Depth and Results per Partition) clearly so users understand how to adjust them to get better results.
	- Build Credibility: Transparently list the data sources (TreeOfLife-200M) and infrastructure (Ohio Supercomputer Center) to establish trust.

Writing Styles: 
Adheres to Ohio State FEP Technical Communication principles:
	- Audience-Centered: Translates engineering jargon into lay terms.
	- Concise
	- Precise
	- Direct
```

# How BioCLIP Image Search Works

**Purpose:** This guide explains how the BioCLIP image search tool identifies visually similar nature images from the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M). It outlines the underlying technology and defines the search controls available to the user.

## Introduction

The BioCLIP Search tool allows you to upload a photo of a plant, animal, or insect to find scientifically similar examples from a database of 200 million images. The system uses a specialized artificial intelligence model called **[BioCLIP-2](https://imageomics.github.io/bioclip-2/)**. Unlike standard search engines, BioCLIP-2 analyzes visual biological traits—such as evolutionary features and morphology—rather than relying on text keywords.

## Image Embedding: The "Biological Fingerprint"

Computers cannot "see" images like humans do. Instead, this tool uses BioCLIP-2 to translate your image into data.

- **The Process:** When you upload an image, BioCLIP-2 converts it into a unique list of numbers, known as an **embedding**. Think of this as a **digital fingerprint** that captures the biological essence of the organism.
    
- **The Match:** The system compares your image's fingerprint to the fingerprints of 200 million other images. Because BioCLIP-2 was trained on the "Tree of Life," it understands biological relationships. BioCLIP-2 can also pick up emergent features such as life stages and habitats. 

## The Search Process: A Library Analogy

Checking 200 million images one by one would take too long. To solve this, we organize the data like a giant library using the [FAISS](https://faiss.ai/index.html) software. 

1. **The Sections (Clusters):** We have divided the 200 million images into over **65,000 distinct groups** in high-dimensionality space. Think of these as specific sections in a library, such as "Weevils," "Ferns," or "Finches."
    
2. **The Selection:** When you search, the system first identifies which groups your image likely belongs to.
    
3. **The Retrieval:** It then searches _only_ within those specific groups to find the closest matches. This two-step process ensures results are both fast and scientifically relevant.

## Search Controls

You can adjust two settings in the application to refine your search results. These controls balance speed against thoroughness.

### 1. Search Depth (`nprobe`)

- **Definition:** This determines how many "library sections" (clusters) the system searches.
    
- **Low Setting (Faster):** The system checks only the single most likely group. This is very fast but might miss a match if the organism is categorized in a closely related neighboring group.
    
- **High Setting (Thorough):** The system broadens its search to check multiple related groups. This increases the chance of finding the best match but takes slightly longer.
    
The chart below illustrates why increasing the Search Depth is often necessary.

![Search Depth Demonstration](search_depth_demo.png)

**Left (Low Depth):** The user's image (Red X) falls just inside the blue region. However, the _true_ best match (Green Star) sits just across the border in the neighboring grey region. Because the system is set to look in only one region, it hits a "hard wall" and misses the best match.

**Right (High Depth):** By increasing the search depth, the system is allowed to check neighboring regions. It successfully crosses the border and finds the Green Star.

**Note on Complexity:** This visualization uses a simple flat (2-dimensional) map for clarity. The actual BioCLIP image search system operates in a massive **768-dimensional space**. In that complex environment, "borders" are much harder to define, making it even more important to check multiple neighboring groups to ensure you don't miss a relevant result hiding "just around the corner."
### 2. Top N Results (`top_n`)

- **Definition:** This setting controls the number of final "best matches" displayed.
    
- **Outcome:** Increasing this number provides a wider variety of results; decreasing it shows you only the very closest matches. **Higher numbers take longer.** Retrieving more results requires the system to fetch more image files and biological details (metadata) from the disk storage space. Asking for 100 images will be slower than asking for 10.