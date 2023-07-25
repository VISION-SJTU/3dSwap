## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = '../pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = '../checkpoints'
embedding_base_dir = '../embeddings'
styleclip_output_dir = './StyleCLIP_results'
experiments_output_dir = '../output'

## Input info
### Input dir, where the images reside
input_data_path = '../dataset/aligned'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'barcelona'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = '../editings/interfacegan_directions/age.pt'
interfacegan_smile = '../editings/interfacegan_directions/smile.pt'
interfacegan_rotation = '../editings/interfacegan_directions/rotation.pt'
ffhq_pca = '../editings/ganspace_pca/ffhq_pca.pt'

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
