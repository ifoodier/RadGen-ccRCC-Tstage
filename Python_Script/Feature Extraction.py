import copy
import traceback
import pandas as pd
import SimpleITK as sitk


from radiomics import featureextractor

def calculate_features(modality=None, is_3d=True):
    # Example
    fn_ori_nrrd = "image.nrrd"
    fn_voi_nrrd = "mask.nrrd"
    try:
        ori_img = sitk.ReadImage(fn_ori_nrrd)
        ori_image_array = sitk.GetArrayFromImage(ori_img)
        shape = ori_image_array.shape
        if len(shape) > 3:
            ori_img = sitk.VectorIndexSelectionCast(ori_img, 0, sitk.sitkUInt16)
        ogi_direction = ori_img.GetDirection()
    except Exception as e:
        print(e)
        return None

    try:
        voi_img = sitk.ReadImage(fn_voi_nrrd)
        voi_image_array = sitk.GetArrayFromImage(voi_img)
        shape = voi_image_array.shape
        if len(shape) > 3:
            voi_img = sitk.VectorIndexSelectionCast(voi_img, 0, sitk.sitkUInt16)
        voi_direction = voi_img.GetDirection()
        if ogi_direction != voi_direction:
            voi_img.SetDirection(ogi_direction)
    except Exception:
        print(traceback.format_exc())
        return None

    if ori_img.GetDimension() == 3:
        ori_origin = ori_img.GetOrigin()
        voi_origin = voi_img.GetOrigin()
        if ori_origin[2] != voi_origin[2]:
            voi_data = sitk.GetArrayFromImage(voi_img)
            depth = ori_img.GetDepth()
            for i in range(depth / 2):
                slice_temp = copy.deepcopy(voi_data[i, :, :])
                voi_data[i, :, :] = copy.deepcopy(voi_data[depth - 1 - i, :, :])
                voi_data[depth - 1 - i, :, :] = copy.deepcopy(slice_temp)
            voi_img = sitk.GetImageFromArray(voi_data)
            voi_img.CopyInformation(ori_img)

    settings = {
        "geometryTolerance": 0.0001,
        "normalize": True,
        "binWidth": 25
    }

    voxelSettings = {
        'kernelRadius': 3,
        'maskedKernel': True,
        'initValue': 0,
       'voxelBatch': 100,
    }
    settings.update(voxelSettings)

    print(f"feature calculate setting: {settings}")

    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        voi_img = sitk.BinaryThreshold(voi_img, lowerThreshold=1, upperThreshold=255)
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()
        features = extractor.execute(ori_img, voi_img, voxelBased=True)

        import matplotlib.pyplot as plt
        for i in range(39):
            plt.figure()
            plt.imshow(sitk.GetArrayFromImage(features['original_firstorder_Maximum'][i]), cmap='jet')
            plt.colorbar()

            output_path = 'new_original_firstorder_Maximum{}.png'.format(str(i))
            print(output_path)
            plt.savefig(output_path)

        return features
    except Exception:
        print(traceback.format_exc())
        return None
if __name__ == '__main__':
    features = calculate_features(modality="CT")
    print(len(features))
    features_dict = features.items()
    features_df = pd.DataFrame(list(features_dict)).T
    features_df.columns = features_df.iloc[0, :]
    features_df.drop(index=[0], inplace=True)
    features_df.to_csv('features.csv')



