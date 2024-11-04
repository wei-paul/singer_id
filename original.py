def wav2mfcc(file_path, max_pad_len=100):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=wave, sr=sr)

    # If MFCC is longer than max_pad_len, truncate it
    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]
    # If MFCC is shorter than max_pad_len, pad it
    else:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')

    return mfcc


def prepare_dataset(data_path):
    X = []
    y = []
    labels = []

    for i, artist in enumerate(os.listdir(data_path)):
        artist_path = os.path.join(data_path, artist)
        labels.append(artist)
        print(f"Processing artist: {artist}")
        file_count = 0
        for audio_file in os.listdir(artist_path):
            file_path = os.path.join(artist_path, audio_file)
            mfcc = wav2mfcc(file_path)
            X.append(mfcc)
            y.append(i)
            file_count += 1
        print(f"Processed {file_count} files for {artist}")

    print(f"Total artists processed: {len(labels)}")
    print(f"Total files processed: {len(X)}")
    return np.array(X), np.array(y), labels


# Prepare the dataset
X, y, labels = prepare_dataset('./train')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reshape for CNN
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# One-hot encode the labels
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

# Print shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train_hot.shape)
