function w = Huffman(r, P, S)
%
% Function : Huffman coding
% input    : r --- the number of the source symbols
%            P --- the probability distribution of source symbols
%            S --- the number of source symbols si
% output   : w --- the Huffman codewords wi corresponding to si
%
format long;
if(r == 2)
    w{S(1)} = '0';
    w{S(2)} = '1';
else
    [P_descend, idx] = sort(P, 'descend');                      % sort {Pi} in descending order
    S = S(idx);                                                 % sort {Si} according to the order of {Pi}
    P = [P_descend(1:r-2)  (P_descend(r-1) + P_descend(r))];    % update P
    lchild = S(r - 1);                                          % left child
    rchild = S(r);                                              % right child
    S = S(1:r-1);                                               % update S
    w = Huffman(r-1, P, S);                         % recursion
    w{rchild} = [w{lchild} '1'];
    w{lchild} = [w{lchild} '0'];
    
end