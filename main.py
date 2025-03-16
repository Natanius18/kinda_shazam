import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import cosine, pdist, squareform

# ANSI escape code for underlining text
UNDERLINE = '\033[4m'
RESET = '\033[0m'
PLOT_TOP_10 = False

db_path = "db"
test_path = "test"


# Function to compute an audio fingerprint and display its spectrogram
def get_fingerprint(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Average the spectrogram to reduce dimensionality
    fingerprint = np.mean(mel_spec_db, axis=1)
    return fingerprint


# Create a heatmap of the similarity between all songs in the database
def plot_similarity_heatmap(fingerprints, distance_type='cosine'):
    names = list(fingerprints.keys())
    vectors = np.array([fingerprints[name] for name in names])

    if distance_type == 'cosine':
        dists = pdist(vectors, metric='cosine')
    elif distance_type == 'linear':
        dists = pdist(vectors, metric='euclidean')
    else:
        raise ValueError("Invalid distance_type")

    dist_matrix = squareform(dists)

    # Кластеризація для сортування (але без дендрограми)
    linkage_matrix = linkage(dists, method='average')
    sorted_idx = leaves_list(linkage_matrix)

    sorted_matrix = dist_matrix[sorted_idx][:, sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    plt.figure(figsize=(20, 16))
    sns.heatmap(sorted_matrix, xticklabels=sorted_names, yticklabels=sorted_names, annot=True, fmt=".4f" if distance_type == 'cosine' else ".0f",
                cmap=sns.color_palette("Oranges", as_cmap=True).reversed())
    plt.title(f"Heatmap of the similarity between all songs ({distance_type} distance)")
    plt.xlabel("Songs")
    plt.ylabel("Songs")
    plt.tight_layout()
    plt.show()


# Display spectrograms for all songs in the database folder
def plot_all_spectrograms():
    num_files = len(os.listdir(db_path))
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files))

    if num_files == 1:
        axes = [axes]

    for ax, filename in zip(axes, os.listdir(db_path)):
        file_path = os.path.join(db_path, filename)
        y, sr = librosa.load(file_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(f"{os.path.splitext(os.path.basename(file_path))[0]}", fontsize=10)
        ax.set_xlabel("Time (sec)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)

    plt.tight_layout()
    plt.show()


# plot_all_spectrograms()


# Compare two audio files by displaying waveforms and different spectrograms
def plot_comparison(test_audio, matched_audio, main_title):
    test_y, test_sr = librosa.load(test_audio, sr=None)
    matched_y, matched_sr = librosa.load(matched_audio, sr=None)

    # STFT spectrograms
    test_stft = librosa.stft(test_y)
    test_stft_db = librosa.amplitude_to_db(np.abs(test_stft), ref=np.max)

    matched_stft = librosa.stft(matched_y)
    matched_stft_db = librosa.amplitude_to_db(np.abs(matched_stft), ref=np.max)

    # Mel spectrograms
    test_mel = librosa.feature.melspectrogram(y=test_y, sr=test_sr)
    test_mel_db = librosa.power_to_db(test_mel, ref=np.max)

    matched_mel = librosa.feature.melspectrogram(y=matched_y, sr=matched_sr)
    matched_mel_db = librosa.power_to_db(matched_mel, ref=np.max)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(main_title, fontsize=14)

    # Waveform plot
    axes[0][0].plot(np.linspace(0, len(test_y) / test_sr, len(test_y)), test_y, color='blue', linewidth=0.06)
    axes[0][0].set_title(f"Waveform (Test: {os.path.basename(test_audio)})", fontsize=10)

    axes[0][1].plot(np.linspace(0, len(matched_y) / matched_sr, len(matched_y)), matched_y, color='green', linewidth=0.06)
    axes[0][1].set_title(f"Waveform (Match: {os.path.basename(matched_audio)})", fontsize=10)

    # STFT spectrogram
    img1 = librosa.display.specshow(test_stft_db, sr=test_sr, x_axis='time', y_axis='log', ax=axes[1][0])
    axes[1][0].set_title("STFT Spectrogram (Test)")
    fig.colorbar(img1, ax=axes[1][0], format='%+2.0f dB')

    img2 = librosa.display.specshow(matched_stft_db, sr=matched_sr, x_axis='time', y_axis='log', ax=axes[1][1])
    axes[1][1].set_title("STFT Spectrogram (Match)")
    fig.colorbar(img2, ax=axes[1][1], format='%+2.0f dB')

    # Mel spectrogram
    img3 = librosa.display.specshow(test_mel_db, sr=test_sr, x_axis='time', y_axis='mel', ax=axes[2][0])
    axes[2][0].set_title("Mel-Spectrogram (Test)")
    fig.colorbar(img3, ax=axes[2][0], format='%+2.0f dB')

    img4 = librosa.display.specshow(matched_mel_db, sr=matched_sr, x_axis='time', y_axis='mel', ax=axes[2][1])
    axes[2][1].set_title("Mel-Spectrogram (Match)")
    fig.colorbar(img4, ax=axes[2][1], format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.show()


# Create a database mapping from fingerprint to song name
fingerprints_db = {}
for filename in os.listdir(db_path):
    file_path = os.path.join(db_path, filename)
    fingerprint = get_fingerprint(file_path)
    fingerprints_db[os.path.splitext(filename)[0]] = fingerprint

# Plot similarity heatmap for all fingerprints in the database
# plot_similarity_heatmap(fingerprints_db, distance_type='cosine')
# plot_similarity_heatmap(fingerprints_db, distance_type='linear')

# Search for matches for test files
for test_file in os.listdir(test_path):
    test_file_path = os.path.join(test_path, test_file)
    test_fingerprint = get_fingerprint(test_file_path)

    distances = {'linear': {}, 'cosine': {}}

    # Compare the test fingerprint with each fingerprint in the database
    for song, fp in fingerprints_db.items():
        # Calculate linear (Euclidean) distance
        linear_dist = np.linalg.norm(test_fingerprint - fp)
        distances['linear'][song] = linear_dist

        # Calculate cosine distance
        cosine_dist = cosine(test_fingerprint, fp)
        distances['cosine'][song] = cosine_dist

    # Sort matches by the smallest distance and select the top-10 for both distance metrics
    sorted_linear_matches = sorted(distances['linear'].items(), key=lambda x: x[1])[:10]
    sorted_cosine_matches = sorted(distances['cosine'].items(), key=lambda x: x[1])[:10]

    # Determine the best match based on linear and cosine distances
    best_linear_match = sorted_linear_matches[0][0]
    best_cosine_match = sorted_cosine_matches[0][0]
    matched_audio_path = os.path.join(db_path, f"{best_linear_match}.mp3")
    plot_comparison(test_file_path, matched_audio_path, "Linear Distance")
    if best_linear_match != best_cosine_match:
        matched_audio_path = os.path.join(db_path, f"{best_cosine_match}.mp3")
        plot_comparison(test_file_path, matched_audio_path, "Cosine Distance")

    print(f"File {UNDERLINE}{test_file}{RESET} is most similar to {UNDERLINE}{best_linear_match}{RESET} "
          f"by linear distance: {sorted_linear_matches[0][1]:.5f}")
    print(f"File {UNDERLINE}{test_file}{RESET} is most similar to {UNDERLINE}{best_cosine_match}{RESET} "
          f"by cosine distance: {sorted_cosine_matches[0][1]:.5f}\n")

    # Visualize the top-10 matches for both distance metrics
    if PLOT_TOP_10:
        songs_linear = [match[0] for match in sorted_linear_matches]
        scores_linear = [match[1] for match in sorted_linear_matches]

        plt.figure(figsize=(10, 6))
        plt.bar(songs_linear, scores_linear, color='skyblue')
        plt.ylabel("Linear Distance", fontsize=10)
        plt.xlabel("Song", fontsize=10)
        plt.xticks(rotation=20, ha='right', fontsize=6)
        plt.title(f"Top-10 Linear Matches for {test_file}", fontsize=12)
        plt.show()

        songs_cosine = [match[0] for match in sorted_cosine_matches]
        scores_cosine = [match[1] for match in sorted_cosine_matches]

        plt.figure(figsize=(10, 6))
        plt.bar(songs_cosine, scores_cosine, color='lightcoral')
        plt.ylabel("Cosine Distance", fontsize=10)
        plt.xlabel("Song", fontsize=10)
        plt.xticks(rotation=20, ha='right', fontsize=6)
        plt.title(f"Top-10 Cosine Matches for {test_file}", fontsize=12)
        plt.show()
