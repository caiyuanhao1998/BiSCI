clear;clc;
res_path = 'results/Test_result.mat';
load(res_path);

psnr_total=0.0;
ssim_total=0.0;
psnr_list = zeros(10,1);
ssim_list = zeros(10,1);
quality_scores = zeros(6,10);
for i=1:10
    Z = squeeze(pred(i,:,:,:));
    Z = double(Z);
    S = squeeze(truth(i,:,:,:));
    S = double(S);
    
    Z(Z>1.0) = 1.0;
    Z(Z<0.0) = 0.0;
    
    [psnr, rmse, ergas, sam, uiqi, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1);
    
    quality_scores(1,i) = psnr;
    quality_scores(2,i) = rmse;
    quality_scores(3,i) = ergas;
    quality_scores(4,i) = sam;
    quality_scores(5,i) = uiqi;
    quality_scores(6,i) = ssim;
    psnr_list(i,1) = psnr;
    ssim_list(i,1) = ssim;
    
    psnr_total = psnr_total+psnr;
    ssim_total = ssim_total+ssim;
end
psnr = mean(psnr_list);
ssim = mean(ssim_list);
fprintf('The PNSR=%f\n',psnr);
fprintf('The SSIM=%f\n',ssim);

