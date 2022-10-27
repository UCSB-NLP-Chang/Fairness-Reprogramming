# The following codes are mainly from https://github.com/Trusted-AI/AIF360
import copy
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from logging import warning

import numpy as np
import pandas as pd


class Dataset(ABC):
    """Abstract base class for data_utils."""

    @abstractmethod
    def __init__(self, **kwargs):
        self.metadata = kwargs.pop('metadata', dict()) or dict()
        self.metadata.update({
            'transformer': '{}.__init__'.format(type(self).__name__),
            'params': kwargs,
            'previous': []
        })
        self.validate_dataset()

    def validate_dataset(self):
        """Error checking and type validation."""
        pass

    def copy(self, deepcopy=False):
        """Convenience method to return a copy of this dataset.
        Args:
            deepcopy (bool, optional): :func:`~copy.deepcopy` this dataset if
                `True`, shallow copy otherwise.
        Returns:
            Dataset: A new dataset with fields copied from this object and
            metadata set accordingly.
        """
        cpy = copy.deepcopy(self) if deepcopy else copy.copy(self)
        # preserve any user-created fields
        cpy.metadata = cpy.metadata.copy()
        cpy.metadata.update({
            'transformer': '{}.copy'.format(type(self).__name__),
            'params': {'deepcopy': deepcopy},
            'previous': [self]
        })
        return cpy

    @abstractmethod
    def export_dataset(self):
        """Save this Dataset to disk."""
        raise NotImplementedError

    @abstractmethod
    def split(self, num_or_size_splits, shuffle=False):
        """Split this dataset into multiple partitions.
        Args:
            num_or_size_splits (array or int): If `num_or_size_splits` is an
                int, *k*, the value is the number of equal-sized folds to make
                (if *k* does not evenly divide the dataset these folds are
                approximately equal-sized). If `num_or_size_splits` is an array
                of type int, the values are taken as the indices at which to
                split the dataset. If the values are floats (< 1.), they are
                considered to be fractional proportions of the dataset at which
                to split.
            shuffle (bool, optional): Randomly shuffle the dataset before
                splitting.
        Returns:
            list(Dataset): Splits. Contains *k* or `len(num_or_size_splits) + 1`
            data_utils depending on `num_or_size_splits`.
        """
        raise NotImplementedError


class StructuredDataset(Dataset):
    """Base class for all structured data_utils.
    A StructuredDataset requires data to be stored in :obj:`numpy.ndarray`
    objects with :obj:`~numpy.dtype` as :obj:`~numpy.float64`.
    Attributes:
        features (numpy.ndarray): Dataset features for each instance.
        labels (numpy.ndarray): Generic label corresponding to each instance
            (could be ground-truth, predicted, cluster assignments, etc.).
        scores (numpy.ndarray): Probability score associated with each label.
            Same shape as `labels`. Only valid for binary labels (this includes
            one-hot categorical labels as well).
        protected_attributes (numpy.ndarray): A subset of `features` for which
            fairness is desired.
        feature_names (list(str)): Names describing each dataset feature.
        label_names (list(str)): Names describing each label.
        protected_attribute_names (list(str)): A subset of `feature_names`
            corresponding to `protected_attributes`.
        privileged_protected_attributes (list(numpy.ndarray)): A subset of
            protected attribute values which are considered privileged from a
            fairness perspective.
        unprivileged_protected_attributes (list(numpy.ndarray)): The remaining
            possible protected attribute values which are not included in
            `privileged_protected_attributes`.
        instance_names (list(str)): Indentifiers for each instance. Sequential
            integers by default.
        instance_weights (numpy.ndarray):  Weighting for each instance. All
            equal (ones) by default. Pursuant to standard practice in social
            science data, 1 means one person or entity. These weights are hence
            person or entity multipliers (see:
            https://www.ibm.com/support/knowledgecenter/en/SS3RA7_15.0.0/com.ibm.spss.modeler.help/netezza_decisiontrees_weights.htm)
            These weights *may not* be normalized to sum to 1 across the entire
            dataset, rather the nominal (default) weight of each entity/record
            in the data is 1. This is similar in spirit to the person weight in
            census microdata samples.
            https://www.census.gov/programs-surveys/acs/technical-documentation/pums/about.html
        ignore_fields (set(str)): Attribute names to ignore when doing equality
            comparisons. Always at least contains `'metadata'`.
        metadata (dict): Details about the creation of this dataset. For
            example::
                {
                    'transformer': 'Dataset.__init__',
                    'params': kwargs,
                    'previous': None
                }
    """

    def __init__(self, df, label_names, protected_attribute_names,
                 instance_weights_name=None, scores_names=None,
                 unprivileged_protected_attributes=None,
                 privileged_protected_attributes=None, metadata=None):
        """
        Args:
            df (pandas.DataFrame): Input DataFrame with features, labels, and
                protected attributes. Values should be preprocessed
                to remove NAs and make all data numerical. Index values are
                taken as instance names.
            label_names (iterable): Names of the label columns in `df`.
            protected_attribute_names (iterable): List of names corresponding to
                protected attribute columns in `df`.
            instance_weights_name (optional): Column name in `df` corresponding
                to instance weights. If not provided, `instance_weights` will be
                all set to 1.
            unprivileged_protected_attributes (optional): If not provided, all
                but the highest numerical value of each protected attribute will
                be considered not privileged.
            privileged_protected_attributes (optional): If not provided, the
                highest numerical value of each protected attribute will be
                considered privileged.
            metadata (optional): Additional metadata to append.
        Raises:
            TypeError: Certain fields must be np.ndarrays as specified in the
                class description.
            ValueError: ndarray shapes must match.
        """
        if privileged_protected_attributes is None:
            privileged_protected_attributes = []
        if unprivileged_protected_attributes is None:
            unprivileged_protected_attributes = []
        if scores_names is None:
            scores_names = []
        if df is None:
            raise TypeError("Must provide a pandas DataFrame representing "
                            "the data (features, labels, protected attributes)")
        if df.isna().any().any():
            raise ValueError("Input DataFrames cannot contain NA values.")
        try:
            df = df.astype(np.float64)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("DataFrame values must be numerical.")

        # Convert all column names to strings
        df.columns = df.columns.astype(str).tolist()
        label_names = list(map(str, label_names))
        protected_attribute_names = list(map(str, protected_attribute_names))

        self.feature_names = [n for n in df.columns if n not in label_names
                              and (not scores_names or n not in scores_names)
                              and n != instance_weights_name]
        self.label_names = label_names
        self.features = df[self.feature_names].values.copy()
        self.labels = df[self.label_names].values.copy()
        self.instance_names = df.index.astype(str).tolist()

        if scores_names:
            self.scores = df[scores_names].values.copy()
        else:
            self.scores = self.labels.copy()

        df_prot = df.loc[:, protected_attribute_names]
        self.protected_attribute_names = df_prot.columns.astype(str).tolist()
        self.protected_attributes = df_prot.values.copy()

        # Infer the privileged and unprivileged values in not provided
        if unprivileged_protected_attributes and privileged_protected_attributes:
            self.unprivileged_protected_attributes = unprivileged_protected_attributes
            self.privileged_protected_attributes = privileged_protected_attributes
        else:
            self.unprivileged_protected_attributes = [
                np.sort(np.unique(df_prot[attr].values))[:-1]
                for attr in self.protected_attribute_names]
            self.privileged_protected_attributes = [
                np.sort(np.unique(df_prot[attr].values))[-1:]
                for attr in self.protected_attribute_names]

        if instance_weights_name:
            self.instance_weights = df[instance_weights_name].values.copy()
        else:
            self.instance_weights = np.ones_like(self.instance_names,
                                                 dtype=np.float64)

        # always ignore metadata and ignore_fields
        self.ignore_fields = {'metadata', 'ignore_fields'}

        # sets metadata
        super(StructuredDataset, self).__init__(df=df, label_names=label_names,
                                                protected_attribute_names=protected_attribute_names,
                                                instance_weights_name=instance_weights_name,
                                                unprivileged_protected_attributes=unprivileged_protected_attributes,
                                                privileged_protected_attributes=privileged_protected_attributes,
                                                metadata=metadata)

    def subset(self, indexes):
        """ Subset of dataset based on position
        Args:
            indexes: iterable which contains row indexes
        Returns:
            `StructuredDataset`: subset of dataset based on indexes
        """
        # convert each element of indexes to string
        indexes_str = [self.instance_names[i] for i in indexes]
        subset = self.copy()
        subset.instance_names = indexes_str
        subset.features = self.features[indexes]
        subset.labels = self.labels[indexes]
        subset.instance_weights = self.instance_weights[indexes]
        subset.protected_attributes = self.protected_attributes[indexes]
        subset.scores = self.scores[indexes]
        return subset

    def __eq__(self, other):
        """Equality comparison for StructuredDatasets.
        Note: Compares all fields other than those specified in `ignore_fields`.
        """
        if not isinstance(other, StructuredDataset):
            return False

        def _eq(x, y):
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                return np.all(x == y)
            elif isinstance(x, list) and isinstance(y, list):
                return len(x) == len(y) and all(_eq(xi, yi) for xi, yi in zip(x, y))
            return x == y

        return all(_eq(self.__dict__[k], other.__dict__[k])
                   for k in self.__dict__.keys() if k not in self.ignore_fields)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        # return repr(self.metadata)
        return str(self)

    def __str__(self):
        df, _ = self.convert_to_dataframe()
        df.insert(0, 'instance_weights', self.instance_weights)
        highest_level = ['instance weights'] + \
                        ['features'] * len(self.feature_names) + \
                        ['labels'] * len(self.label_names)
        middle_level = [''] + \
                       ['protected attribute'
                        if f in self.protected_attribute_names else ''
                        for f in self.feature_names] + \
                       [''] * len(self.label_names)
        lowest_level = [''] + self.feature_names + [''] * len(self.label_names)
        df.columns = pd.MultiIndex.from_arrays(
            [highest_level, middle_level, lowest_level])
        df.index.name = 'instance names'
        return str(df)

    # TODO: *_names checks
    def validate_dataset(self):
        """Error checking and type validation.
        Raises:
            TypeError: Certain fields must be np.ndarrays as specified in the
                class description.
            ValueError: ndarray shapes must match.
        """
        super(StructuredDataset, self).validate_dataset()

        # =========================== TYPE CHECKING ============================
        for f in [self.features, self.protected_attributes, self.labels,
                  self.scores, self.instance_weights]:
            if not isinstance(f, np.ndarray):
                raise TypeError("'{}' must be an np.ndarray.".format(f.__name__))

        # convert ndarrays to float64
        self.features = self.features.astype(np.float64)
        self.protected_attributes = self.protected_attributes.astype(np.float64)
        self.labels = self.labels.astype(np.float64)
        self.instance_weights = self.instance_weights.astype(np.float64)

        # =========================== SHAPE CHECKING ===========================
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape((-1, 1))
        try:
            self.scores.reshape(self.labels.shape)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("'scores' should have the same shape as 'labels'.")
        if not self.labels.shape[0] == self.features.shape[0]:
            raise ValueError(
                "Number of labels must match number of instances:"
                "\n\tlabels.shape = {}\n\tfeatures.shape = {}".format(
                    self.labels.shape, self.features.shape))
        if not self.instance_weights.shape[0] == self.features.shape[0]:
            raise ValueError(
                "Number of weights must match number of instances:"
                "\n\tinstance_weights.shape = {}\n\tfeatures.shape = {}".format(
                    self.instance_weights.shape, self.features.shape))

        # =========================== VALUE CHECKING ===========================
        if np.any(np.logical_or(self.scores < 0., self.scores > 1.)):
            warning("'scores' has no well-defined meaning out of range [0, 1].")

        for i in range(len(self.privileged_protected_attributes)):
            priv = set(self.privileged_protected_attributes[i])
            unpriv = set(self.unprivileged_protected_attributes[i])
            # check for duplicates
            if priv & unpriv:
                raise ValueError(
                    "'privileged_protected_attributes' and 'unprivileged_protected_attributes' "
                    "should not share any common elements:\n\tBoth contain {} for feature {}".format(
                        list(priv & unpriv), self.protected_attribute_names[i]))
            # check for unclassified values
            if not set(self.protected_attributes[:, i]) <= (priv | unpriv):
                raise ValueError(
                    "All observed values for protected attributes should be designated as "
                    "either privileged or unprivileged: \n\t{} not designated for feature {}".format(
                        list(set(self.protected_attributes[:, i]) - (priv | unpriv)),
                        self.protected_attribute_names[i]))
            # warn for unobserved values
            if not (priv | unpriv) <= set(self.protected_attributes[:, i]):
                warning("{} listed but not observed for feature {}".format(
                    list((priv | unpriv) - set(self.protected_attributes[:, i])),
                    self.protected_attribute_names[i]))

    @contextmanager
    def temporarily_ignore(self, *fields):
        """Temporarily add the fields provided to `ignore_fields`.
        To be used in a `with` statement. Upon completing the `with` block,
        `ignore_fields` is restored to its original value.
        Args:
            *fields: Additional fields to ignore for equality comparison within
                the scope of this context manager, e.g.
                `temporarily_ignore('features', 'labels')`. The temporary
                `ignore_fields` attribute is the union of the old attribute and
                the set of these fields.
        Examples:
            >>> sd = StructuredDataset(...)
            >>> modified = sd.copy()
            >>> modified.labels = sd.labels + 1
            >>> assert sd != modified
            >>> with sd.temporarily_ignore('labels'):
            >>>     assert sd == modified
            >>> assert 'labels' not in sd.ignore_fields
        """
        old_ignore = deepcopy(self.ignore_fields)
        self.ignore_fields |= set(fields)
        try:
            yield
        finally:
            self.ignore_fields = old_ignore

    def align_datasets(self, other):
        """Align the other dataset features, labels and protected_attributes to
        this dataset.
        Args:
            other (StructuredDataset): Other dataset that needs to be aligned
        Returns:
            StructuredDataset: New aligned dataset
        """

        if (set(self.feature_names) != set(other.feature_names) or
                set(self.label_names) != set(other.label_names) or
                set(self.protected_attribute_names)
                != set(other.protected_attribute_names)):
            raise ValueError(
                "feature_names, label_names, and protected_attribute_names "
                "should match between this and other dataset.")

        # New dataset
        new = other.copy()

        # re-order the columns of the new dataset
        feat_inds = [new.feature_names.index(f) for f in self.feature_names]
        label_inds = [new.label_names.index(f) for f in self.label_names]
        prot_inds = [new.protected_attribute_names.index(f)
                     for f in self.protected_attribute_names]

        new.features = new.features[:, feat_inds]
        new.labels = new.labels[:, label_inds]
        new.scores = new.scores[:, label_inds]
        new.protected_attributes = new.protected_attributes[:, prot_inds]

        new.privileged_protected_attributes = [
            new.privileged_protected_attributes[i] for i in prot_inds]
        new.unprivileged_protected_attributes = [
            new.unprivileged_protected_attributes[i] for i in prot_inds]
        new.feature_names = deepcopy(self.feature_names)
        new.label_names = deepcopy(self.label_names)
        new.protected_attribute_names = deepcopy(self.protected_attribute_names)

        return new

    # TODO: Should we store the protected attributes as a separate dataframe
    def convert_to_dataframe(self, de_dummy_code=False, sep='=',
                             set_category=True):
        """Convert the StructuredDataset to a :obj:`pandas.DataFrame`.
        Args:
            de_dummy_code (bool): Performs de_dummy_coding, converting dummy-
                coded columns to categories. If `de_dummy_code` is `True` and
                this dataset contains mappings for label and/or protected
                attribute values to strings in the `metadata`, this method will
                convert those as well.
            sep (char): Separator between the prefix in the dummy indicators and
                the dummy-coded categorical levels.
            set_category (bool): Set the de-dummy coded features to categorical
                type.
        Returns:
            (pandas.DataFrame, dict):
                * `pandas.DataFrame`: Equivalent dataframe for a dataset. All
                  columns will have only numeric values. The
                  `protected_attributes` field in the dataset will override the
                  values in the `features` field.
                * `dict`: Attributes. Will contain additional information pulled
                  from the dataset such as `feature_names`, `label_names`,
                  `protected_attribute_names`, `instance_names`,
                  `instance_weights`, `privileged_protected_attributes`,
                  `unprivileged_protected_attributes`. The metadata will not be
                  returned.
        """
        df = pd.DataFrame(np.hstack((self.features, self.labels)),
                          columns=self.feature_names + self.label_names,
                          index=self.instance_names)
        df.loc[:, self.protected_attribute_names] = self.protected_attributes

        # De-dummy code if necessary
        if de_dummy_code:
            df = self._de_dummy_code_df(df, sep=sep, set_category=set_category)
            if 'label_maps' in self.metadata:
                for i, label in enumerate(self.label_names):
                    df[label] = df[label].replace(self.metadata['label_maps'][i])
            if 'protected_attribute_maps' in self.metadata:
                for i, prot_attr in enumerate(self.protected_attribute_names):
                    df[prot_attr] = df[prot_attr].replace(
                        self.metadata['protected_attribute_maps'][i])

        # Attributes
        attributes = {
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "protected_attribute_names": self.protected_attribute_names,
            "instance_names": self.instance_names,
            "instance_weights": self.instance_weights,
            "privileged_protected_attributes": self.privileged_protected_attributes,
            "unprivileged_protected_attributes": self.unprivileged_protected_attributes
        }

        return df, attributes

    def export_dataset(self, export_metadata=False):
        """
        Export the dataset and supporting attributes
        TODO: The preferred file format is HDF
        """

        if export_metadata:
            raise NotImplementedError("The option to export metadata has not been implemented yet")

        return None

    def import_dataset(self, import_metadata=False):
        """ Import the dataset and supporting attributes
            TODO: The preferred file format is HDF
        """

        if import_metadata:
            raise NotImplementedError("The option to import metadata has not been implemented yet")
        return None

    def split(self, num_or_size_splits, shuffle=False, seed=None):
        """Split this dataset into multiple partitions.
        Args:
            num_or_size_splits (list or int): If `num_or_size_splits` is an
                int, *k*, the value is the number of equal-sized folds to make
                (if *k* does not evenly divide the dataset these folds are
                approximately equal-sized). If `num_or_size_splits` is an array
                of type int, the values are taken as the indices at which to
                split the dataset. If the values are floats (< 1.), they are
                considered to be fractional proportions of the dataset at which
                to split.
            shuffle (bool, optional): Randomly shuffle the dataset before
                splitting.
            seed (int or array_like): Takes the same argument as
                :func:`numpy.random.seed()`.
        Returns:
            list: Splits. Contains *k* or `len(num_or_size_splits) + 1`
            data_utils depending on `num_or_size_splits`.
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        n = self.features.shape[0]
        if isinstance(num_or_size_splits, list):
            num_folds = len(num_or_size_splits) + 1
            if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
                num_or_size_splits = [int(x * n) for x in num_or_size_splits]
        else:
            num_folds = num_or_size_splits

        order = list(np.random.permutation(n) if shuffle else range(n))
        folds = [self.copy() for _ in range(num_folds)]

        features = np.array_split(self.features[order], num_or_size_splits)
        labels = np.array_split(self.labels[order], num_or_size_splits)
        scores = np.array_split(self.scores[order], num_or_size_splits)
        protected_attributes = np.array_split(self.protected_attributes[order],
                                              num_or_size_splits)
        instance_weights = np.array_split(self.instance_weights[order],
                                          num_or_size_splits)
        instance_names = np.array_split(np.array(self.instance_names)[order],
                                        num_or_size_splits)
        for fold, feats, labs, scors, prot_attrs, inst_wgts, inst_name in zip(
                folds, features, labels, scores, protected_attributes, instance_weights,
                instance_names):
            fold.features = feats
            fold.labels = labs
            fold.scores = scors
            fold.protected_attributes = prot_attrs
            fold.instance_weights = inst_wgts
            fold.instance_names = list(map(str, inst_name))
            fold.metadata = fold.metadata.copy()
            fold.metadata.update({
                'transformer': '{}.split'.format(type(self).__name__),
                'params': {'num_or_size_splits': num_or_size_splits,
                           'shuffle': shuffle},
                'previous': [self]
            })

        return folds

    @staticmethod
    def _de_dummy_code_df(df, sep="=", set_category=False):
        feature_names_dum_d, feature_names_nodum = \
            StructuredDataset._parse_feature_names(df.columns)
        df_new = pd.DataFrame(index=df.index,
                              columns=feature_names_nodum + list(feature_names_dum_d.keys()))

        for fname in feature_names_nodum:
            df_new[fname] = df[fname].values.copy()

        for fname, vl in feature_names_dum_d.items():
            for v in vl:
                df_new.loc[df[fname + sep + str(v)] == 1, fname] = str(v)

        if set_category:
            for fname in feature_names_dum_d.keys():
                df_new[fname] = df_new[fname].astype('category')

        return df_new

    @staticmethod
    def _parse_feature_names(feature_names, sep="="):
        feature_names_dum_d = defaultdict(list)
        feature_names_nodum = list()
        for fname in feature_names:
            if sep in fname:
                fname_dum, v = fname.split(sep, 1)
                feature_names_dum_d[fname_dum].append(v)
            else:
                feature_names_nodum.append(fname)

        return feature_names_dum_d, feature_names_nodum


class BinaryLabelDataset(StructuredDataset):
    """Base class for all structured data_utils with binary labels."""

    def __init__(self, favorable_label=1., unfavorable_label=0., **kwargs):
        """
        Args:
            favorable_label (float): Label value which is considered favorable
                (i.e. "positive").
            unfavorable_label (float): Label value which is considered
                unfavorable (i.e. "negative").
            **kwargs: StructuredDataset arguments.
        """
        self.favorable_label = float(favorable_label)
        self.unfavorable_label = float(unfavorable_label)

        super(BinaryLabelDataset, self).__init__(**kwargs)

    def validate_dataset(self):
        """Error checking and type validation.
        Raises:
            ValueError: `labels` must be shape [n, 1].
            ValueError: `favorable_label` and `unfavorable_label` must be the
                only values present in `labels`.
        """
        # fix scores before validating
        if np.all(self.scores == self.labels):
            self.scores = (self.scores == self.favorable_label).astype(np.float64)

        super(BinaryLabelDataset, self).validate_dataset()

        # =========================== SHAPE CHECKING ===========================
        # Verify if the labels are only 1 column
        if self.labels.shape[1] != 1:
            raise ValueError("BinaryLabelDataset only supports single-column "
                             "labels:\n\tlabels.shape = {}".format(self.labels.shape))

        # =========================== VALUE CHECKING ===========================
        # Check if the favorable and unfavorable labels match those in the dataset
        if not set(self.labels.ravel()) <= {self.favorable_label, self.unfavorable_label}:
            raise ValueError("The favorable and unfavorable labels provided do "
                             "not match the labels in the dataset.")


class StandardDataset(BinaryLabelDataset):
    """Base class for every :obj:`BinaryLabelDataset` provided out of the box by
    aif360.
    It is not strictly necessary to inherit this class when adding custom
    data_utils but it may be useful.
    This class is very loosely based on code from
    https://github.com/algofairness/fairness-comparison.
    """

    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=None, features_to_keep=None,
                 features_to_drop=None, custom_preprocessing=None,
                 metadata=None):
        """
        Subclasses of StandardDataset should perform the following before
        calling `super().__init__`:
            1. Load the dataframe from a raw file.
        Then, this class will go through a standard preprocessing routine which:
            2. (optional) Performs some dataset-specific preprocessing (e.g.
               renaming columns/values, handling missing data).
            3. Drops unrequested columns (see `features_to_keep` and
               `features_to_drop` for details).
            4. Drops rows with NA values.
            5. Creates a one-hot encoding of the categorical variables.
            6. Maps protected attributes to binary privileged/unprivileged
               values (1/0).
            7. Maps labels to binary favorable/unfavorable labels (1/0).
        Args:
            df (pandas.DataFrame): DataFrame on which to perform standard
                processing.
            label_name: Name of the label column in `df`.
            favorable_classes (list or function): Label values which are
                considered favorable or a boolean function which returns `True`
                if favorable. All others are unfavorable. Label values are
                mapped to 1 (favorable) and 0 (unfavorable) if they are not
                already binary and numerical.
            protected_attribute_names (list): List of names corresponding to
                protected attribute columns in `df`.
            privileged_classes (list(list or function)): Each element is
                a list of values which are considered privileged or a boolean
                function which return `True` if privileged for the corresponding
                column in `protected_attribute_names`. All others are
                unprivileged. Values are mapped to 1 (privileged) and 0
                (unprivileged) if they are not already numerical.
            instance_weights_name (optional): Name of the instance weights
                column in `df`.
            categorical_features (optional, list): List of column names in the
                DataFrame which are to be expanded into one-hot vectors.
            features_to_keep (optional, list): Column names to keep. All others
                are dropped except those present in `protected_attribute_names`,
                `categorical_features`, `label_name` or `instance_weights_name`.
                Defaults to all columns if not provided.
            features_to_drop (optional, list): Column names to drop. *Note: this
                overrides* `features_to_keep`.
            custom_preprocessing (function): A function object which
                acts on and returns a DataFrame (f: DataFrame -> DataFrame). If
                `None`, no extra preprocessing is applied.
            metadata (optional): Additional metadata to append.
        """
        # 2. Perform dataset-specific preprocessing
        if features_to_drop is None:
            features_to_drop = []
        if features_to_keep is None:
            features_to_keep = []
        if categorical_features is None:
            categorical_features = []

        if custom_preprocessing:
            df = custom_preprocessing(df)

        # 3. Drop unrequested columns
        features_to_keep = features_to_keep or df.columns.tolist()
        keep = (set(features_to_keep) | set(protected_attribute_names)
                | set(categorical_features) | {label_name})
        if instance_weights_name:
            keep |= {instance_weights_name}
        df = df[sorted(keep - set(features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            warning("Missing Data: {} rows removed from {}.".format(count,
                                                                    type(self).__name__))
        df = dropped

        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

        # 6. Map protected attributes to privileged/unprivileged
        privileged_protected_attributes = []
        unprivileged_protected_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            privileged_values = [1.]
            unprivileged_values = [0.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

            privileged_protected_attributes.append(
                np.array(privileged_values, dtype=np.float64))
            unprivileged_protected_attributes.append(
                np.array(unprivileged_values, dtype=np.float64))

        # 7. Make labels binary
        favorable_label = 1.
        unfavorable_label = 0.
        if callable(favorable_classes):
            df[label_name] = df[label_name].apply(favorable_classes)
        elif np.issubdtype(df[label_name], np.number) and len(set(df[label_name])) == 2:
            # labels are already binary; don't change them
            favorable_label = favorable_classes[0]
            unfavorable_label = set(df[label_name]).difference(favorable_classes).pop()
        else:
            # find all instances which match any of the favorable classes
            pos = np.logical_or.reduce(np.equal.outer(favorable_classes,
                                                      df[label_name].to_numpy()))
            df.loc[pos, label_name] = favorable_label
            df.loc[~pos, label_name] = unfavorable_label

        super(StandardDataset, self).__init__(df=df, label_names=[label_name],
                                              protected_attribute_names=protected_attribute_names,
                                              privileged_protected_attributes=privileged_protected_attributes,
                                              unprivileged_protected_attributes=unprivileged_protected_attributes,
                                              instance_weights_name=instance_weights_name,
                                              scores_names=[scores_name] if scores_name else [],
                                              favorable_label=favorable_label,
                                              unfavorable_label=unfavorable_label, metadata=metadata)


default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}, {1.0: 'Male', 0.0: 'Female'}]
}


class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.
    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, label_name='income-per-year', favorable_classes=None, protected_attribute_names=None,
                 privileged_classes=None, instance_weights_name=None, categorical_features=None, features_to_keep=None,
                 features_to_drop=None, na_values=None, custom_preprocessing=None, metadata=None):

        if features_to_keep is None:
            features_to_keep = []
        if features_to_drop is None:
            features_to_drop = ['fnlwgt']
        if na_values is None:
            na_values = ['?']
        if metadata is None:
            metadata = default_mappings
        if categorical_features is None:
            categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                    'native-country']
        if privileged_classes is None:
            privileged_classes = [['White'], ['Male']]
        if protected_attribute_names is None:
            protected_attribute_names = ['race', 'sex']
        if favorable_classes is None:
            favorable_classes = ['>50K', '>50K.']

        train_path = os.path.join(os.getcwd(), ".data", 'adult', 'adult.data')
        test_path = os.path.join(os.getcwd(), ".data", 'adult', 'adult.test')

        # as given by adult.names
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income-per-year']
        try:
            train = pd.read_csv(train_path, header=None, names=column_names,
                                skipinitialspace=True, na_values=na_values)
            test = pd.read_csv(test_path, header=0, names=column_names,
                               skipinitialspace=True, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
                os.path.abspath(__file__), '../..', '..', 'data', 'raw', 'adult'))))
            import sys
            sys.exit(1)

        df = pd.concat([test, train], ignore_index=True)

        super(AdultDataset, self).__init__(df=df, label_name=label_name,
                                           favorable_classes=favorable_classes,
                                           protected_attribute_names=protected_attribute_names,
                                           privileged_classes=privileged_classes,
                                           instance_weights_name=instance_weights_name,
                                           categorical_features=categorical_features,
                                           features_to_keep=features_to_keep,
                                           features_to_drop=features_to_drop,
                                           custom_preprocessing=custom_preprocessing, metadata=metadata)


def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x // 10 * 10)

        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x <= 5:
                return '<6'
            elif x >= 13:
                return '>12'
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        # Limit age range
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # Rename income variable
        df['Income Binary'] = df['income-per-year']
        df['Income Binary'] = df['Income Binary'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['Income Binary'] = df['Income Binary'].replace(to_replace='<=50K.', value='<=50K', regex=True)

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Income Binary'] == '<=50K']
            df_1 = df[df['Income Binary'] == '>50K']
            df_0 = df_0.sample(int(sub_samp / 2))
            df_1 = df_1.sample(int(sub_samp / 2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['Age (decade)', 'Education Years', 'sex', 'race']
    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    Y_features = ['Income Binary']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['Age (decade)', 'Education Years']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}

    return AdultDataset(
        label_name=Y_features[0],
        favorable_classes=['>50K', '>50K.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features + Y_features + D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                               for x in D_features]},
        custom_preprocessing=custom_preprocessing)
