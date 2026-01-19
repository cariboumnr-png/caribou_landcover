'''Data block class.'''

# standard imports
import dataclasses
import json
import math
import typing
# third party imports
import numpy

@dataclasses.dataclass
class Data:
    '''Simple dataclass for block-wise image/label data.'''

    label: numpy.ndarray=dataclasses.field(init=False)
    label_masked: numpy.ndarray=dataclasses.field(init=False)
    image: numpy.ndarray=dataclasses.field(init=False)
    image_dem_padded: numpy.ndarray=dataclasses.field(init=False)
    image_normalized: numpy.ndarray=dataclasses.field(init=False)
    valid_mask: numpy.ndarray=dataclasses.field(init=False)
    meta: dict=dataclasses.field(init=False)

    def __repr__(self) -> str:
        lines = ['Block summary\n']
        lines.append('-' * 70)
        # iterate attributes from the dataclass
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, numpy.ndarray):
                lines.append(f'{field.name}: ')
                lines.append(f'Array shape: {value.shape}')
                lines.append(f'Array dtype: {value.dtype}')
                lines.append('-' * 70)
            elif isinstance(value, dict):
                lines.append(f'{field.name}: ')
                lines.append(f'{json.dumps(value, indent=4)}')
                lines.append('-' * 70)
            else:
                lines.append(f'{field.name}: ')
                lines.append(f'{value}')
                lines.append('-' * 70)
        # return lines
        return '\n'.join(lines)

    def validate(self, skip_attr: str | None=None):
        '''Validate if all attr has been populated.'''
        for field in dataclasses.fields(self):
            if skip_attr is not None and field.name == skip_attr:
                setattr(self, skip_attr, numpy.empty([1])) # place holder
            if not hasattr(self, field.name):
                raise ValueError(f"{field.name} has not been populated yet")

class DataBlock():
    '''Data block compiled from input label and image rasters.'''

    def __init__(self):
        '''Can be instantiate without arguments.'''
        # init data
        self.data = Data()

    def load_from_rasters(
            self,
            lbl_arr: numpy.ndarray,
            img_arr: numpy.ndarray,
            dem_padded: numpy.ndarray,
            meta: dict[str, typing.Any]
        ) -> 'DataBlock':
        '''Create a new block from raw input rasters.'''

        # assertions - both 3 dims and the same H and W
        assert len(lbl_arr.shape) == 3 and len(img_arr.shape) == 3
        assert lbl_arr.shape[-2] == img_arr.shape[-2]

        # assign self attrs
        self.data.meta = meta
        self.data.label = lbl_arr.astype(numpy.uint8)[0] # (1, H, W) tp (H, W)
        self.data.image = img_arr.astype(numpy.float32) # remote sensing default
        self.data.image_dem_padded = dem_padded.astype(numpy.float32) # padded

        # process data sequence
        # image bands related
        self._image_add_spec_indices()
        self._image_add_topo_metrics()
        # labels related
        self._labels_get_structure()
        self._label_count_classes()
        # block-wise
        self._block_get_valid_mask()
        self.block_get_image_stats()
        # metadata
        self.__reorder_meta()
        # sanity check self.data
        self.data.validate(skip_attr='image_normalized') # to be populated later

        # return self to allow chained calls
        return self

    def load_from_npz(self, fpath: str) -> 'DataBlock':
        '''Load data from a pre-existing .npz file.'''

        # load and set attributes
        loaded = numpy.load(fpath, allow_pickle=True)
        for key in loaded:
            value = loaded[key]
            # 'meta' is a pickled dict, convert from 0-d array
            if key == 'meta' and isinstance(value, numpy.ndarray):
                value = value.item()
            # set value
            try:
                setattr(self.data, key, value)
            except AttributeError:
                continue # don't anything here yet

        # return self to allow chained calls
        return self

    def normalize_image(self, global_stats: dict, **kwargs) -> tuple:
        '''Normalize image bands using provided global band stats.'''

        # assertion
        assert len(global_stats) == self.data.image.shape[0]

        # parse args
        update_norm = kwargs.get('update_norm', ())
        band_assignment = self.data.meta.get('band_assignment', {})

        # init data attribute, inherit dtype float32
        self.data.image_normalized = numpy.empty_like(self.data.image)

        # normalize each band
        for i, (band, stats) in enumerate(global_stats.items()):
            # if to be skiped
            if not list(band_assignment.keys())[i] in update_norm and update_norm:
                continue
            # sanity check - dict keys from band_0
            assert band.lstrip('band_') == str(i)
            # get global stats from input
            g_mean = stats['current_mean']
            g_std = stats['std'] if stats['std'] != 0 else 1
            # get image band and replace invalid pixels with global mean
            img_band = self.data.image[i]
            img_band = numpy.where(self.data.valid_mask, img_band, g_mean)
            # normalize band
            self.data.image_normalized[i] = (img_band - g_mean) / g_std

        # sanity check the minmax of normalized image
        mmin = self.data.image_normalized.min().item()
        mmax = self.data.image_normalized.max().item()
        # return
        return mmin, mmax

    def save_npz(self, fpath: str, compress: bool=True) -> None:
        '''Save block data as an `.npz` file. Will overwrite.'''

        # sanity check
        assert fpath.endswith('.npz')
        # directly write self.data to npz file
        save_data = vars(self.data)
        # save file - allow pickle to write meta dict
        if compress:
            numpy.savez_compressed(fpath, allow_pickle=True, **save_data)
        else:
            numpy.savez(fpath, allow_pickle=True, **save_data)

    def _labels_get_structure(self):
        '''Prep the hierarchy of labels according to metadata.'''

        # get kwargs
        label_nodata = self.data.meta.get('label_nodata', 0) # ignore 0 -> safe
        label1_to_ignore = self.data.meta.get('label1_to_ignore', [])
        ignore_label = self.data.meta.get('ignore_label', 255)
        label1_reclass = self.data.meta.get('label1_reclass_map', {})

        # fill invalid pixels with ignore label
        label1_to_ignore = list(label1_to_ignore) # avoid modifying the meta
        label1_to_ignore.append(label_nodata)
        label1_to_ignore = [x for x in label1_to_ignore if x is not None]
        layer1_mask = ~numpy.isin(self.data.label, label1_to_ignore)
        layer1_valid = numpy.where(layer1_mask, self.data.label, ignore_label)

        # collection of final layers
        fn_stack = [layer1_valid] # layer1 as the first element of the list

        # iterate through layer1 classes in reclass map
        for band_num, classes in label1_reclass.items():
            # mask to the current layer 1 class (as the layer2 subclasses)
            _mask = numpy.isin(self.data.label, classes)
            # in-place reclass relevant pixels in layer1
            fn_stack[0][_mask] = int(band_num)
            # create a layer 2 array for current layer1 class
            layer2_new = numpy.where(_mask, self.data.label, ignore_label)
            # reclass from 1 to n
            for i, cls in enumerate(classes, 1):
                layer2_new[layer2_new == cls] = int(i)
            # append the new layer 2 array to the stack
            fn_stack.append(layer2_new)

        # stack all the arrays and assign to attr
        self.data.label_masked = numpy.stack(fn_stack, axis=0)

        # add metadata
        self.data.meta['valid_pixel_ratio'] = {
            'layer1': numpy.sum(layer1_mask) / layer1_valid.size
        }
        for i in range(len(label1_reclass)):
            _valid = fn_stack[i + 1] != ignore_label
            self.data.meta['valid_pixel_ratio'].update({
                f'layer2_{i + 1}': numpy.sum(_valid) / layer1_valid.size
            })

    def _label_count_classes(self) -> None:
        '''Count present label values and calculate entropy.'''

        # write to following entries for meta
        self.data.meta['label_count'] = {}
        self.data.meta['label_entropy'] = {}

        # supposed number of classes for each label layer
        n_classes = [
            self.data.meta['label1_num_classes'], # count of original label
            len(self.data.meta['label1_reclass_map']) # count of reclassed lyr1
        ]
        n_classes.extend([
            len(v) for v in self.data.meta['label1_reclass_map'].values()
        ]) # counts of lyr2 groups

        # all arrays to be counted
        lyrs = numpy.concatenate(
            [self.data.label[None, :, :], self.data.label_masked], axis=0
        ) # e.g., (7, 256, 256)

        # sanity check
        assert len(n_classes) == len(lyrs)

        # iteration
        for i, band in enumerate(lyrs):
            # count unique values for each label layer
            label_unique = numpy.arange(1, n_classes[i] + 1) # start from 1
            filtered = band[numpy.isin(band, label_unique)]
            uniques, counts = numpy.unique(filtered, return_counts=True)

            # convert to list. avoid using arr.tolist()
            uu = [int(_) for _ in uniques]
            cc = [int(_) for _ in counts]

            # shannon entropy
            ent = 0.0
            for _ in cc:
                p = _ / sum(cc)
                ent -= p * math.log2(p)

            # assign zero count to no_show classes
            counts = []
            for _ in range(n_classes[i]):
                idx = _ + 1
                if idx in uu:
                    count_idx = uu.index(idx)
                    counts.append(cc[count_idx])
                else:
                    counts.append(0)

            # add to metadata
            if i == 0:
                self.data.meta['label_count']['original_label'] = counts
                self.data.meta['label_entropy']['original_label'] = ent
            elif i == 1:
                self.data.meta['label_count']['layer1'] = counts
                self.data.meta['label_entropy']['layer1'] = ent
            else:
                self.data.meta['label_count'][f'layer2_{i - 1}'] = counts
                self.data.meta['label_entropy'][f'layer2_{i - 1}'] = ent

    def _image_add_spec_indices(self) -> None:
        '''Add spectral indices using loaded Landsat bands.'''

        # skip if already created
        if self.data.meta.get('spectral_indices_added', False):
            return

        # retrieve from meta
        spec_bands = self.data.meta.get('band_assignment', {})
        nodata = self.data.meta.get('image_nodata', numpy.nan)

        # assertions
        assert all(k in spec_bands for k in ['red', 'nir', 'swir1', 'swir2'])

        # mask off nodata pixels to avoid overflow
        # 32 bit increased to 64 bit float
        red = self.__mask(self.data.image[spec_bands['red']], nodata)
        nir = self.__mask(self.data.image[spec_bands['nir']], nodata)
        swir1 = self.__mask(self.data.image[spec_bands['swir1']], nodata)
        swir2 = self.__mask(self.data.image[spec_bands['swir2']], nodata)

        # add data to class
        self.data.meta['spectral_indices_added'] = ['NDVI', 'NDMI', 'NBR']
        n = len(self.data.meta['band_assignment'])
        self.data.meta['band_assignment'].update({
            'NDVI': n, 'NDMI': n + 1, 'NBR': n + 2
        })
        add_indices = numpy.stack([
            self.__ndvi(nir, red, nodata),
            self.__ndmi(nir, swir1, nodata),
            self.__nbr(nir, swir2, nodata)
        ]).astype(numpy.float32)
        self.data.image = numpy.append(self.data.image, add_indices, axis=0)

    def _image_add_topo_metrics(self) -> None:
        '''Add topographical metrics to the image array.'''

        # skip if already created
        if self.data.meta.get('topo_metrics_added', False):
            return

        # get vars
        pad = self.data.meta['dem_pad']
        nodata = self.data.meta.get('image_nodata', numpy.nan)
        max_h, max_w = self.data.image_dem_padded.shape
        # sanity check
        assert self.data.image[0].shape == (max_h - 2 * pad, max_w - 2 * pad)

        # prep metrics to add
        slope = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        cos_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        sin_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        tpi = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)

        # iterate through pixels from the original block in padded dem
        for y in range(pad, max_h - pad):
            for x in range(pad, max_w - pad):
                # slope and aspect - 8 neighbors, radius 1
                pxs = self.__get_px_group(self.data.image_dem_padded, x, y, 1)
                slope[y - pad, x - pad], cos_a[y - pad, x - pad], \
                    sin_a[y - pad, x - pad] = self.__slope_n_aspect(pxs, nodata)
                # tpi - 224 neighbors, radius 7
                pxs =  self.__get_px_group(self.data.image_dem_padded, x, y, 7)
                tpi[y  - pad, x - pad] = self.__tpi(pxs, nodata)

        # add to metadata
        self.data.meta['topo_metrics_added'] = [
            'slope', 'cos_aspect', 'sin_aspect', 'tpi'
        ]
        n = len(self.data.meta['band_assignment'])
        self.data.meta['band_assignment'].update({
            'slope': n, 'cos_aspect': n + 1, 'sin_aspect': n + 2,
            'tpi': n + 3
        })

        # add to self.image
        to_add = numpy.stack([slope, cos_a, sin_a, tpi], axis=0)
        self.data.image = numpy.append(self.data.image, to_add, axis=0)

    def _block_get_valid_mask(self):
        '''Get a valid mask for the whole block.'''

        #
        img_nodata = self.data.meta.get('image_nodata', numpy.nan)
        ignore_label = self.data.meta.get('ignore_label', 255)

        # required - locs where label layer1 valid (not 255)
        lbl_lyr1_valid = self.data.label_masked[0] != ignore_label # (256, 256)
        # required - locs where image all bands valid
        invalid_image = numpy.isnan(self.data.image) | \
            numpy.isclose(self.data.image, img_nodata) # + nodata
        img_all_valid = ~numpy.any(invalid_image, axis=0) # shape (256, 256)
        # final mask
        self.data.valid_mask = lbl_lyr1_valid & img_all_valid # (256, 256)
        # add ratio to meta
        self.data.meta['valid_pixel_ratio'].update({
            'image': float((numpy.sum(img_all_valid) / img_all_valid.size)),
            'block': float((numpy.sum(self.data.valid_mask) / img_all_valid.size))
        })

    def block_get_image_stats(self):
        '''Per block stats for later aggregation using Welford's.'''

        # parse
        image_nodata = self.data.meta.get('image_nodata', numpy.nan)
        # create dict for block stats
        self.data.meta['block_image_stats'] = {}
        #
        for i, band in enumerate(self.data.image):
            # get where pixel is not valid
            if isinstance(image_nodata, float) and numpy.isnan(image_nodata):
                mask = numpy.isnan(band)
            else:
                mask = numpy.isclose(band, image_nodata)
            # inverse to get valid pixels
            valid = band[~mask]
            # 3 block stats to be done: nb, mb, m2b
            num = valid.size
            mean = numpy.mean(valid)
            mean_sq = numpy.sum((valid - mean) ** 2)
            # give to self.image_stats
            self.data.meta['block_image_stats'][f'band_{i}'] = {
                'count': int(num), 'mean': float(mean), 'm2': float(mean_sq)
            }

    def __reorder_meta(self):
        '''Reorder meta dict keys.'''

        # desired key order
        new_order = [
            # block
            'block_name',
            'block_shape',
            'valid_pixel_ratio',
            # label
            'label_nodata',
            'ignore_label',
            'label1_num_classes',
            'label1_to_ignore',
            'label1_class_name',
            'label1_reclass_map',
            'label1_reclass_name',
            'label_count',
            'label_entropy',
            # image
            'image_nodata',
            'dem_pad',
            'band_assignment',
            'spectral_indices_added',
            'topo_metrics_added',
            'block_image_stats'
        ]

        # assertion
        assert set(new_order) == set(self.data.meta.keys()), \
            f'{new_order} \n {self.data.meta.keys()}'

        # reorder
        self.data.meta = {k: self.data.meta[k] for k in new_order}

    # static calculation methods
    # spectral indices related
    @staticmethod
    def __mask(band, nodata):
        # avoid overflow downstream
        return numpy.ma.masked_where(
            numpy.isclose(band, nodata), band.astype(numpy.float64)
        )

    @staticmethod
    def __ndvi(nir, red, nodata):
        out = (nir - red) / (nir + red)
        return out.filled(nodata)

    @staticmethod
    def __ndmi(nir, swir1, nodata):
        out = (nir - swir1) / (nir + swir1)
        return out.filled(nodata)

    @staticmethod
    def __nbr(nir, swir2, nodata):
        out = (nir - swir2) / (nir + swir2)
        return out.filled(nodata)

    # topographical metrics related
    @staticmethod
    def __get_px_group(arr, x, y, np):
        return arr[slice(y - np, y + np + 1), slice(x - np, x + np + 1)]

    @staticmethod
    def __slope_n_aspect(arr, nodata):
        # all 9 cells need to have a valid value (Horn's)
        if numpy.any(numpy.isclose(arr, nodata)) or \
            numpy.isnan(arr).any() or \
                numpy.isinf(arr).any():
            return nodata, nodata, nodata
        # calculation
        dz_dx = (
            (arr[0, 2] + 2 * arr[1, 2] + arr[2, 2]) - \
            (arr[0, 0] + 2 * arr[1, 0] + arr[2, 0])
        ) / 8.0
        dz_dy = (
            (arr[2, 0] + 2 * arr[2, 1] + arr[2, 2]) - \
            (arr[0, 0] + 2 * arr[0, 1] + arr[0, 2])
        ) / 8.0
        # calculate slope
        slope = numpy.sqrt(dz_dx ** 2 + dz_dy ** 2)
        # calculate aspect angle in radians
        aspect_rad = numpy.arctan2(dz_dy, -dz_dx)
        if aspect_rad < 0:
            aspect_rad += 2 * numpy.pi  # Normalize to [0, 2π]
        # compute cosine and sine of aspect
        cos_aspect = numpy.cos(aspect_rad)
        sin_aspect = numpy.sin(aspect_rad)
        # return
        return slope, cos_aspect, sin_aspect

    @staticmethod
    def __tpi(arr, nodata):
        # topographical position index
        h, w = arr.shape
        c_row, c_col = h // 2, w // 2
        centre = arr[c_row, c_col]
        # centre pixel is nodata
        if numpy.isclose(centre, nodata):
            return nodata
        masked = numpy.ma.masked_where(numpy.isclose(arr, nodata), arr)
        # all is nodata except centre
        if masked.count() == 1:
            return nodata
        # valid arr
        return centre - (masked.sum() - centre) / (masked.count() - 1)
