from ammf.core.feature_extraction_phase.bev_vgg import BevVgg
from ammf.core.feature_extraction_phase.bev_vgg_pyramid import BevVggPyr

from ammf.core.feature_extraction_phase.img_vgg import ImgVgg
from ammf.core.feature_extraction_phase.img_vgg_pyramid import ImgVggPyr


def get_extractor(extractor_config):

    extractor_type = extractor_config.WhichOneof('feature_extractor')

    # BEV feature extractors
    if extractor_type == 'bev_vgg':
        return BevVgg(extractor_config.bev_vgg)
    elif extractor_type == 'bev_vgg_pyr':
        return BevVggPyr(extractor_config.bev_vgg_pyr)

    # Image feature extractors
    elif extractor_type == 'img_vgg':
        return ImgVgg(extractor_config.img_vgg)
    elif extractor_type == 'img_vgg_pyr':
        return ImgVggPyr(extractor_config.img_vgg_pyr)

    #TODO: modify this to make DepVgg work or comment this
    # Dep feature extractors
    # elif extractor_type == 'dep_vgg':
    #     return DepVgg(extractor_config.dep_vgg)
    # elif extractor_type == 'dep_vgg_pyr':
    #     return DepVggPyr(extractor_config.dep_vgg_pyr)
    

    return ValueError('Invalid feature extractor type', extractor_type)
