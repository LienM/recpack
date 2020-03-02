class DataM:
    def __init__(self, values_sp_m, timestamps_sp_m=None):
        self._values = values_sp_m
        self._timestamps = timestamps_sp_m
        self._validate_matrices_shape()

    def _validate_matrices_shape(self):
        if self._timestamps is not None:
            assert self._values.shape == self._timestamps.shape

    @property
    def values(self):
        self._validate_matrices_shape()
        return self._values

    @property
    def timestamps(self):
        self._validate_matrices_shape()
        if self._timestamps is None:
            raise AttributeError("timestamps is None, and should not be used")
        return self._timestamps

    @property
    def shape(self):
        return self._values.shape

    def apply_mask(self, mask_sp_m):
        """
        Pointwise multiple the matrices with the mask provided. Shape of mask should match shape of object.
        Will edit matrices in place.

        :param mask_sp_m: The mask to apply to the data matrices. 
                          A mask is a 1 and 0 matrix which will be pointwise multiplated with data matrices.
        :type mask_sp_m: `scipy.sparse.csr_matrix`
        """
        # Make sure shapes are correct
        assert mask_sp_m.shape == self._values.shape
        self._validate_matrices_shape()

        self._values = self._values.multiply(mask_sp_m)
        if self._timestamps is not None:
            self._timestamps = self._timestamps.multiply(mask_sp_m)

    def copy(self):
        c_values = self._values.copy()
        c_timestamps = None if self._timestamps is None else self._timestamps.copy()

        return DataM(c_values, c_timestamps)
