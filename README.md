# Shazam (kinda 🤭)

## Overview

Kinda Shazam is a simple music recognition system built using Python and the `librosa` library. It works by computing
the audio fingerprint of songs and comparing test audio clips against a database of known tracks to identify matches.
The project includes visualizations of spectrograms and comparison metrics to analyze the recognition process.

## Features

- Extracts Mel-spectrogram-based fingerprints from audio files.
- Matches test clips against a database using similarity metrics.
- Displays spectrograms for all stored and test audio files.
- Provides a ranked list of the best matches for each test clip.
- Supports waveform, STFT and Mel-spectrogram visualizations.


## Usage
> [!IMPORTANT]
> Ensure you have Python 3 installed along with the following dependencies:
> ```
> pip install librosa numpy matplotlib scipy
> ```
### Step 1: Prepare Audio Files

- Place 20-second clips of known songs in the `db/` folder.
- Place test clips in the `test/` folder.

### Step 2: Run the Script

Execute the main script `main.py` to analyze and match test audio clips:

### Step 3: View Results

- The script will display spectrograms of the database songs.
- It will attempt to identify each test file and print the best matches.
- A visualization of the top 10 closest matches for each test file is shown.

## How It Works

1. **Audio Fingerprinting**: Converts audio into Mel-spectrogram representations and extracts fingerprints.
2. **Similarity Comparison**: Uses distance metrics (e.g., Euclidean distance and cosine similarity) to compare test fingerprints against the
   database.
3. **Visualization**: Generates spectrograms and ranked match lists to evaluate recognition performance.

## Distance Metrics

To compare fingerprints, different distance measures are used:

### Euclidean Distance

Euclidean distance is the straight-line distance between two points in an N-dimensional space. It is calculated as:

$$ d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2} $$

Lower values indicate more similarity.

### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors and is computed as:

$$ \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| ||\mathbf{B}||} $$

Cosine similarity ranges from -1 (completely opposite) to 1 (identical). Cosine distance is given by:

$$ d_{cosine}(\mathbf{A}, \mathbf{B}) = 1 - \cos(\theta) $$

## Example Output

After running the script, you will see output similar to:

```
File counting-stars-acoustic-cover.mp3 is most similar to OneRepublic – Good Life by linear distance: 41.93989
File counting-stars-acoustic-cover.mp3 is most similar to OneRepublic – Counting Stars by cosine distance: 0.00182
```

## Spectrogram and Match Visualization

Below is an example of the top-10 best matches visualization:
<p align="middle">
  <img src="/plot/linear10.png" width="49%" />
  <img src="/plot/cosine10.png" width="49%" /> 
</p>

For each test song you will see its comparison to the best match: Waveform, STFT spectrogram and Mel-spectrogram. 
If the linear and cosine distances show different results, you will see both visualizations:
<p align="middle">
  <img src="/plot/linear_diagram.png" width="49%" />
  <img src="/plot/cosine_diagram.png" width="49%" /> 
</p>
For example, here we can see that cosine-based approach showed a better result.
<br><br>
Here are spectrograms of the database songs:
<details>
   <summary><i>open/close the diagram</i></summary>
<img src="/plot/db_spectrograms.png" width="60%">
</details>
And their similarities (the lower the number, the more similar two songs are):

<img src="/plot/heatmap_linear.png" width="70%">
<img src="/plot/heatmap_cosine.png" width="70%">



