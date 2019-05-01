%cd matconvnet-1.0-beta23
%run matlab/vl_compilenn ;

% Download a pre-trained CNN from the web (needed once).
%urlwrite(...
%  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
%  'imagenet-googlenet-dag.mat') ;

% Setup MatConvNet.
run matconvnet-1.0-beta23/matlab/vl_setupnn ;

%cd ..
% Load a model and upgrade it to MatConvNet current version.
net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
net.conserveMemory=false;

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

		features = [];
		for i = 3:length(ims)
			% Obtain and preprocess an image.
			im = imread(fullfile(imgdir{k}, 'img', classes{ci}, ims(i).name));
			[d,pid,e]=fileparts(ims(i).name);

			im_ = single(im) ; % note: 255 range
			im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
			im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
			
			net.eval({'data', im_});	

			scores = net.vars(end-2).value;
			scores = squeeze(gather(scores))';
			scores = scores/norm(scores); %l2 norm
			scores = arrayfun(@num2str, scores, 'UniformOutput', false);
			
			features = [features; [pid scores]];
		end
		size(features)
		fname = [classes{ci} ' LeNet.csv'];
		writetable(cell2table(features), fullfile(imgdir{k}, 'descvis', 'img', fname), 'WriteVariableNames', false)

	end

end

