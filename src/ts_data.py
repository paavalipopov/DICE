# pylint: disable=C0103,C0115,C0116,R0913,E1121,C0301
"""Functions for extracting dataset features and labels"""
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from src.settings import DATA_ROOT


def load_ABIDE1(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
    filter_idx: bool = False,
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (569, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 569 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        # filter the data: leave only correct components
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_ABIDE1_869(
    dataset_path: str = DATA_ROOT.joinpath(
        "abide869/ABIDE1_AllData_869Subjects_ICA.npz"
    ),
    indices_path: str = DATA_ROOT.joinpath("abide869/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide869/labels_ABIDE1_869Subjects.csv"),
    filter_idx: bool = False,
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 869 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 295 - time points - data.shape[2]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1

        # filter the data: leave only correct components and the first 156 time points
        # (not all subjects have all 160 time points)
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]
        # 156 - time points - data.shape[2]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_COBRE(
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv"),
    filter_idx: bool = False,
):
    """
    Return COBRE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("COBRE_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (157, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 157 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        # filter the data: leave only correct components
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_FBIRN(
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv"),
    filter_idx: bool = False,
):
    """
    Return FBIRN data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("FBIRN_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (311, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 311 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        # filter the data: leave only correct components
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_OASIS(
    only_first_sessions: bool = True,
    only_two_classes: bool = True,
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz"),
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv"),
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv"),
    filter_idx: bool = False,
):
    """
    Return OASIS data

    Input:
    only_first_sessions: bool = True
    - load only first sessions of each subject
    only_two_classes: bool = True
    - filter all classes except for 0 and 1 (HC and AZ)
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv")
    - path to labels
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv")
    - path to indices of the first sessions

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 2826 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 160 - time points - data.shape[2]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1

        # filter the data: leave only correct components and the first 156 time points
        # (not all subjects have all 160 time points)
        data = data[:, idx, :156]
        # print(data.shape)
        # 53 - components - data.shape[1]
        # 156 - time points - data.shape[2]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    if only_first_sessions:
        # leave only first sessions
        sessions = pd.read_csv(sessions_path, header=None)
        first_session = sessions[0].values - 1

        data = data[first_session, :, :]
        # 912 - sessions - data.shape[0] - only first session
        labels = labels[first_session]

    if only_two_classes:
        # leave subjects of class 0 and 1 only
        filter_array = []
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

        data = data[filter_array, :, :]
        # 2559 - sessions - data.shape[0] - subjects of class 0 and 1
        # 823 - sessions - data.shape[0] - if only first sessions are considered
        labels = labels[filter_array]

    return data, labels


def load_UKB(
    dataset_path: str = "/data/users2/ppopov1/UKB_data/UKB_sex_data.npz",
    indices_path: str = "/data/users2/ppopov1/UKB_data/correct_indices_GSP.csv",
    filter_idx: bool = False,
):
    """
    Return UKB data

    Input:
    dataset_path: str = "/data/users2/ppopov1/UKB_data/UKB_sex_data.npz"
    - path to the dataset with lablels
    indices_path: str = "/data/users2/ppopov1/UKB_data/correct_indices_GSP.csv"
    - path to correct indices/components


    Output:
    features, labels
    """

    features = None
    labels = None
    with np.load(dataset_path) as npzfile:
        features = npzfile["features"]
        labels = npzfile["labels"]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        features = features[:, idx, :]

    return features, labels


def load_BSNIP(
    only_two_classes: bool = True,
    invert_classes: bool = True,
    dataset_path: str = DATA_ROOT.joinpath("bsnip/BSNIP_data.npz"),
    indices_path: str = DATA_ROOT.joinpath("bsnip/correct_indices_GSP.csv"),
    filter_idx: bool = False,
):
    """
    Return BSNIP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("bsnip/BSNIP_data.npz")
    - path to the dataset with lablels
    indices_path: str = DATA_ROOT.joinpath("bsnip/correct_indices_GSP.csv")
    - path to correct indices/components


    Output:
    features, labels
    """

    features = None
    labels = None
    with np.load(dataset_path) as npzfile:
        features = npzfile["features"]
        labels = npzfile["labels"]

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        features = features[:, idx, :]

    if only_two_classes:
        # leave subjects of class 0 and 1 only
        # {"NC": 0, "SZ": 1, "SAD": 2, "BP": 3, "BPnon": 4, "OTH": 5}
        filter_array = []
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

        features = features[filter_array, :, :]
        labels = labels[filter_array]

    if only_two_classes and invert_classes:
        new_labels = []
        for label in labels:
            if label == 0:
                new_labels.append(1)
            else:
                new_labels.append(0)

        labels = np.array(new_labels)

    return features, labels


class TSQuantileTransformer:
    def __init__(self, *args, n_quantiles: int, **kwargs):
        self.n_quantiles = n_quantiles
        self._args = args
        self._kwargs = kwargs
        self.transforms = {}

    def fit(self, features: np.ndarray):
        for i in range(features.shape[1]):
            self.transforms[i] = QuantileTransformer(
                *self._args, n_quantiles=self.n_quantiles, **self._kwargs
            ).fit(features[:, i, :])
        return self

    def transform(self, features: np.ndarray):
        result = np.empty_like(features, dtype=np.int32)
        for i in range(features.shape[1]):
            result[:, i, :] = (
                self.transforms[i].transform(features[:, i, :]) * self.n_quantiles
            ).astype(np.int32)
        return result


def load_time_FBIRN(
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    filter_idx: bool = False,
):
    """
    Return FBIRN normal + inversed time data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv")
    - path to labels

    Output:
    features, labels
    """

    data, _ = load_FBIRN(dataset_path, indices_path)
    inversed_data = np.flip(data, axis=2)

    labels = [0] * data.shape[0]
    labels.extend([1] * data.shape[0])
    labels = np.array(labels)

    data = np.concatenate((data, inversed_data))

    return data, labels


def load_ROI_FBIRN(
    regions: int,
    dataset_path: str = DATA_ROOT.joinpath("fbirn_roi"),
    labels_path: str = DATA_ROOT.joinpath("fbirn_roi/labels_FBIRN_new.csv"),
    filter_idx: bool = False,
):
    """
    Return ROI FBIRN data

    Input:
    regions: 100/200/400/1000 Schaefer atlases
    dataset_path: str = DATA_ROOT.joinpath("fbirn_roi")
    - path to the dataset
    labels_path: str = DATA_ROOT.joinpath("fbirn_roi/labels_FBIRN_new.csv")
    - path to labels

    Output:
    features, labels
    """

    if regions == 100:
        final_dataset_path: str = dataset_path.joinpath(
            f"FBIRN_fMRI_{regions}ShaeferAtlas_onlytimeserieszscored.npz"
        )
    elif regions == 200:
        final_dataset_path: str = dataset_path.joinpath(
            f"FBIRN_fMRI_{regions}ShaeferAtlas_onlytimeserieszscored.npz"
        )
    elif regions == 400:
        final_dataset_path: str = dataset_path.joinpath(
            f"FBIRN_fMRI_{regions}ShaeferAtlas_onlytimeserieszscored.npz"
        )
    elif regions == 1000:
        final_dataset_path: str = dataset_path.joinpath(
            f"FBIRN_fMRI_{regions}ShaeferAtlas_onlytimeserieszscored.npz"
        )
    else:
        raise NotImplementedError()
    # get data
    data = np.load(final_dataset_path)
    # print(data.shape)
    # >>> (311, regions, 160)

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_HCP(
    dataset_path: str = DATA_ROOT.joinpath("hcp/HCP_AllData_sess1.npz"),
    labels_path: str = DATA_ROOT.joinpath("hcp/labels_HCP_Gender.csv"),
    indices_path: str = DATA_ROOT.joinpath("hcp/correct_indices_GSP.csv"),
    filter_idx: bool = False,
):
    """
    Return ICA HCP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("hcp/HCP_AllData_sess1.npz")
    - path to the dataset
    labels_path: str = DATA_ROOT.joinpath("hcp/labels_HCP_Gender")
    - path to labels
    indices_path: str = DATA_ROOT.joinpath("hcp/correct_indices_GSP.csv")
    - path to correct indices/components

    Output:
    features, labels
    """

    # get data
    features = np.load(dataset_path)
    # print(data.shape)
    # >>> (833, 100, 1185)

    if filter_idx:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        features = features[:, idx, :]
        # >>> (833, 53, 1185)

    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int")
    # (833,)

    return features, labels


def load_ROI_HCP(
    dataset_path: str = DATA_ROOT.joinpath("hcp_roi"),
    filter_idx: bool = False,
):
    """
    Return ROI HCP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("hcp_roi")
    - path to the dataset

    Output:
    features, labels
    """

    dataset_path1 = dataset_path.joinpath(
        "HCP_fMRI_200ShaeferAtlas_onlytimeserieszscored.npz"
    )
    dataset_path2 = dataset_path.joinpath(
        "HCP_fMRI_remaining190_200ShaeferAtlas_onlytimeserieszscored.npz"
    )

    label_path1 = dataset_path.joinpath("HCPlabelsIhave.csv")
    label_path2 = dataset_path.joinpath("labels_HCP_remaining_190_subjects.csv")

    # get data
    data1 = np.load(dataset_path1)
    data2 = np.load(dataset_path2)
    # print(data1.shape)
    # print(data2.shape)
    # >>> (752, 200, 1200)
    # >>> (190, 200, 1200)

    labels1 = pd.read_csv(label_path1, header=None)
    labels1 = labels1.values.flatten().astype("int")
    # (752,)

    labels2 = pd.read_csv(label_path2, header=None)
    labels2 = labels2.values.flatten().astype("int")
    # (190,)

    data = np.concatenate((data1, data2))
    labels = np.concatenate((labels1, labels2))

    return data, labels


def load_ROI_ABIDE(
    dataset_path: str = DATA_ROOT.joinpath(
        "abide_roi/ABIDE1_AllData_871Subjects_region_shaefer200_316TP_onlytimeserieszscored.npz"
    ),
    labels_path: str = DATA_ROOT.joinpath("abide_roi/ABIDE1_region_labels_871.csv"),
    filter_idx: bool = False,
):
    """
    Return ROI ABIDE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide_roi/ABIDE1_AllData_871Subjects_region_shaefer200_316TP_onlytimeserieszscored.npz")
    - path to the dataset
    labels_path: str = DATA_ROOT.joinpath("abide_roi/ABIDE1_region_labels_871.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    data = np.load(dataset_path)
    # print(data.shape)
    # >>> (871, 200, 316)

    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1
    # (871,)

    return data, labels


def load_dataset(dataset: str, filter_idx: bool = True):
    """
    Return the dataset defined by type
    """
    if dataset == "oasis":
        data, labels = load_OASIS(filter_idx=filter_idx)
    elif dataset == "abide":
        data, labels = load_ABIDE1(filter_idx=filter_idx)
    elif dataset == "fbirn":
        data, labels = load_FBIRN(filter_idx=filter_idx)
    elif dataset == "cobre":
        data, labels = load_COBRE(filter_idx=filter_idx)
    elif dataset == "abide_869":
        data, labels = load_ABIDE1_869(filter_idx=filter_idx)
    elif dataset == "ukb":
        data, labels = load_UKB(filter_idx=filter_idx)
    elif dataset == "bsnip":
        data, labels = load_BSNIP(filter_idx=filter_idx)
    elif dataset == "time_fbirn":
        data, labels = load_time_FBIRN(filter_idx=filter_idx)
    elif dataset == "fbirn_100":
        data, labels = load_ROI_FBIRN(100, filter_idx=filter_idx)
    elif dataset == "fbirn_200":
        data, labels = load_ROI_FBIRN(200, filter_idx=filter_idx)
    elif dataset == "fbirn_400":
        data, labels = load_ROI_FBIRN(400, filter_idx=filter_idx)
    elif dataset == "fbirn_1000":
        data, labels = load_ROI_FBIRN(1000, filter_idx=filter_idx)
    elif dataset == "hcp":
        data, labels = load_HCP(filter_idx=filter_idx)
    elif dataset == "hcp_roi":
        data, labels = load_ROI_HCP(filter_idx=filter_idx)
    elif dataset == "abide_roi":
        data, labels = load_ROI_ABIDE(filter_idx=filter_idx)
    else:
        raise NotImplementedError()

    return data, labels
