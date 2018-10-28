function w = Shannon(r, P, S)
%
% Function : Shannon coding
% input    : r --- the number of the source symbols
%            P --- the probability distribution of source symbols
%            S --- the number of source symbols si
% output   : w --- the Huffman codewords wi corresponding to si
%
format long;
[P_descend, idx] = sort(P, 'descend');      % sort {Pi} in descending order
F = zeros(1,r);                             % initialize {Fi}
l = zeros(1,r);                             % initialize {li}
for i = 1:r
    F(i) = sum(P_descend(1:i-1));           % cumulative distribution function
    l(i) = ceil(log2(1 / P_descend(i)));    % codelength
    
    F_binary = dec_to_bin(F(i), l(i));      % convert decimal to binary
    
    % take li digit after the dot as the codeword for the source symbol idx(i)
    w{idx(i)} = num2str(F_binary);          % convert to string 
    w{idx(i)} = strrep(w{idx(i)}, ' ','');  % eliminate space
end
    