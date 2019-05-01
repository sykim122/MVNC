%cd matconvnet-1.0-beta23
%run matlab/vl_compilenn ;

% Download a pre-trained CNN from the web (needed once).
%urlwrite(...
%  'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-152-dag.mat', ...
%  'imagenet-resnet-152-dag.mat') ;

% Setup MatConvNet.
run matconvnet-1.0-beta23/matlab/vl_setupnn ;

%cd ..
% Load a model and upgrade it to MatConvNet current version.
net = dagnn.DagNN.loadobj(load('imagenet-resnet-152-dag.mat')) ;
net.conserveMemory=false;

% get classes
path = '../Div400/devset/devsetkeywords/';
imgdir = fullfile(path, 'img');
classes = dir(imgdir);
classes = classes([classes.isdir]);
classes = {classes(3:end).name} ;

for ci = 1:length(classes)
	ims = dir(fullfile(imgdir, classes{ci}));

	features = [];
	for i = 3:length(ims)
		% Obtain and preprocess an image.
		im = imread(fullfile(imgdir, classes{ci}, ims(i).name));
		[d,pid,e]=fileparts(ims(i).name);

		im_ = single(im) ; % note: 255 range
		im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
		im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
		
		net.eval({'data', im_});	

		scores = net.vars(end-2).value;
		scores = squeeze(gather(scores))';
		%scores = scores/norm(scores); %l2 norm
		scores = arrayfun(@num2str, scores, 'UniformOutput', false);
		
		features = [features; [pid scores]];
	end
	size(features)
	fname = [classes{ci} ' Res152net-skipnorm.csv'];
	writetable(cell2table(features), fullfile(path, 'descvis', 'img', fname), 'WriteVariableNames', false)

end

