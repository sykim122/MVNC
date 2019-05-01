%cd matconvnet-1.0-beta23
%run matlab/vl_compilenn ;

% Download a pre-trained CNN from the web (needed once).
%urlwrite(...
%  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', ...
%  'imagenet-vgg-verydeep-19.mat') ;

% Setup MatConvNet.
run matconvnet-1.0-beta23/matlab/vl_setupnn ;

%cd ..
% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-vgg-verydeep-19.mat') ;
net = vl_simplenn_tidy(net) ;

% get classes
imgdir = {'../Div400/devset/devsetkeywordsGPS/',
	'../Div400/testset/testset_keywords/',
	'../Div400/testset/testset_keywordsGPS/'}
for k = 1:length(imgdir)

	classes = dir(fullfile(imgdir{k}, 'img'));
	classes = classes([classes.isdir]);
	classes = {classes(3:end).name} ;
	
	for ci = 1:length(classes)
		ims = dir(fullfile(imgdir{k}, 'img', classes{ci}));

		features = []
		for i = 3:length(ims)
			% Obtain and preprocess an image.
			im = imread(fullfile(imgdir{k}, 'img', classes{ci}, ims(i).name));
			[d,pid,e]=fileparts(ims(i).name);

			im_ = single(im) ; % note: 255 range
			im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
			im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
		
			% Run the CNN.
			res = vl_simplenn(net, im_) ;

			scores = squeeze(gather(res(end-2).x))';
			scores = scores/norm(scores);
			scores = arrayfun(@num2str, scores, 'UniformOutput', false);
			
			features = [features; [pid scores]];
		end
		size(features)
		writetable(cell2table(features), fullfile(imgdir{k},'descvis','img',[classes{ci} ' VGGnet.csv']),'WriteVariableNames',false)

	end
end

