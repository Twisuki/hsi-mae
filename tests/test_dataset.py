"""Tests for HSIDataset."""

import numpy as np
import pytest
import torch

from src.datasets.hsi_dataset import HSIDataset


@pytest.fixture
def synthetic_hsi(tmp_path):
    """Create a synthetic HSI .npy file for testing."""
    # Simulate [H=8, W=8, C=20] HSI cube
    data = np.random.rand(8, 8, 20).astype(np.float32)
    labels = np.random.randint(0, 4, size=(8, 8), dtype=np.int64)
    data_path = tmp_path / "hsi.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(data_path, data)
    np.save(labels_path, labels)
    return str(data_path), str(labels_path)


class TestHSIDataset:
    """Tests for HSIDataset."""

    def test_load_npy(self, synthetic_hsi):
        data_path, _ = synthetic_hsi
        ds = HSIDataset(data_path, unlabeled=True)
        assert ds.data.shape == (8, 8, 20)
        assert ds.num_bands == 20

    def test_normalize(self, synthetic_hsi):
        data_path, _ = synthetic_hsi
        ds = HSIDataset(data_path, unlabeled=True, normalize=True)
        assert ds.data.min() >= 0.0
        assert ds.data.max() <= 1.0

    def test_splits_deterministic(self, synthetic_hsi):
        data_path, labels_path = synthetic_hsi
        ds_train1 = HSIDataset(data_path, labels_path, split="train", train_ratio=0.7)
        ds_train2 = HSIDataset(data_path, labels_path, split="train", train_ratio=0.7)
        assert len(ds_train1) == len(ds_train2)

    def test_getitem_shape(self, synthetic_hsi):
        data_path, _ = synthetic_hsi
        ds = HSIDataset(data_path, unlabeled=True, to_chw=True)
        x, y = ds[0]
        # to_chw=True: [C, 1, 1]
        assert x.shape == (20, 1, 1)
        assert y.shape == (1,)

    def test_unlabeled_mode(self, synthetic_hsi):
        data_path, _ = synthetic_hsi
        ds = HSIDataset(data_path, unlabeled=True)
        _, y = ds[0]
        assert y.item() == 0  # unlabeled → label 0

    def test_labeled_mode(self, synthetic_hsi):
        data_path, labels_path = synthetic_hsi
        ds = HSIDataset(data_path, labels_path, unlabeled=False)
        _, y = ds[0]
        assert 0 <= y.item() < 4

    def test_len(self, synthetic_hsi):
        data_path, _ = synthetic_hsi
        ds = HSIDataset(data_path, unlabeled=True)
        assert len(ds) == 64  # 8*8 pixels

    def test_get_spectral_signature(self, synthetic_hsi):
        data_path, _ = synthetic_hsi
        ds = HSIDataset(data_path, unlabeled=True)
        sig = ds.get_spectral_signature(3, 5)
        assert sig.shape == (20,)

    def test_num_classes(self, synthetic_hsi):
        data_path, labels_path = synthetic_hsi
        ds = HSIDataset(data_path, labels_path, unlabeled=False)
        assert ds.num_classes == 4

    def test_get_full_data(self, synthetic_hsi):
        data_path, labels_path = synthetic_hsi
        ds = HSIDataset(data_path, labels_path)
        data, labels = ds.get_full_data()
        assert data.shape == (8, 8, 20)
        assert labels.shape == (8, 8)

    def test_mat_file_with_wrong_format_raises(self, tmp_path):
        # Unsupported extension should raise
        fake_path = tmp_path / "fake.txt"
        fake_path.write_text("not a hsi file")
        with pytest.raises(ValueError, match="Unsupported"):
            HSIDataset(fake_path)


class TestHSIDatasetLabelsFromMat:
    """Test that labels embedded in .mat can be loaded."""

    def test_labels_embedded_mat(self, tmp_path):
        from scipy.io import savemat

        data = np.random.rand(6, 6, 15).astype(np.float32)
        labels = np.random.randint(0, 3, size=(6, 6), dtype=np.int64)
        mat_path = tmp_path / "embedded.mat"
        savemat(mat_path, {"data": data, "labels": labels})

        ds = HSIDataset(mat_path, mat_key="data", mat_label_key="labels",
                        unlabeled=False)
        assert ds.num_bands == 15
        assert ds.num_classes == 3
