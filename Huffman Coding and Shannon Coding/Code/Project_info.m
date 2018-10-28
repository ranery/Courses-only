%% ============ Course project of Information theory ==============
%
% Project    `  : Implementation of Huffman Coding and Shannon Coding 
% Author        : Haoran_You
% Supervisor    : Yayu_Gao
% E-mail        : RanerY@hust.edu.cn
% Date          : 6.7.2017
%
%
%%  Load Data 
clc
clear
f = [pwd '\Steve_Jobs_Speech.doc'];                   % path
try    
    Word = actxGetRunningServer('Word.Application');  % activate Word
catch    
    Word = actxserver('Word.Application'); 
end;
Word.Visible = 1;                                     % set visible
Document = Word.Documents.Open(f);                    % get the object of this document
Selection = Word.Selection;                           % cursor position
Selection.Start=0;
a=[];
num_char = Document.Range.end;                        % length of the document
ii=0;
while ii <= num_char
    ii = ii + 1;
    a=[a,Selection.text];                             % load data
    Selection.MoveRight;                              % move the cursor to right by one square
end
str = a(1:num_char);                                  % get the content of document
fprintf('The content of the document:\n\n%s\n', str);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

%%  Text Statistics 

length = size(str,2);                                 % get the length of 'str'
pos = zeros(1,100000);                                % initial the position
for n = 1:length
    pos(1,double(str(n))) = pos(1,double(str(n))) + 1;% vote for the characteristics
end
stat = find(pos > 0);                                 % statistic the frequency of characteristics in this document
num_char = 0;
totalnum_char = 0;
disp('statistics of the times of each char¡¯s appearance :')
fprintf('char \t times');
for n = stat
    fprintf('%s \t\t %d\n', char(n), pos(n));
    num_char = num_char + 1;
    totalnum_char = totalnum_char + pos(n);
end
fprintf('The number of different characteristics : %d\n', num_char);
fprintf('The number of total characteristics in the document : %d\n', totalnum_char);

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%%  Information entropy

disp('the probability of each char :')
fprintf('char \t Pr');
P = zeros(1,num_char);                            % initial probability
i = 1;
for n = stat
    P(i) = pos(n) / length;
    fprintf('%s \t\t %d\n', char(n), P(i));
    i = i + 1;
end
fprintf('sum(P) = \n\n \t\t %d\n',sum(P));                                            
entropy = sum(P .* log2(1 ./ P));                 % information entropy
fprintf('The information entropy of the document : %d\n', entropy);

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%%  Huffmam Coding

fprintf('This is Huffman Coding:\n');
S = 1:num_char;
w_H = Huffman(num_char, P, S);                      % Huffman coding 

fprintf('Char \t Pi \t\t\t\t Code');                % print code 
i = 1;
for n = stat
    fprintf('%s \t\t %d \t\t %s\n', char(n), P(i), w_H{i});
    i = i + 1;
end

lenH = cellfun('length',w_H);                        % codelength
lenH_a = lenH .* P;                                
fprintf('The average codelength with Huffman Coding : %d\n', sum(lenH_a));   % average codelength

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%%  Shannon Coding

fprintf('This is Shannon Coding:\n');
w_S = Shannon(num_char, P, S);                      % Shannon Coding

fprintf('Char \t Pi \t\t\t\t Code');                % print code 
i = 1;
for n = stat
    fprintf('%s \t\t %d \t\t %s\n', char(n), P(i), w_S{i});
    i = i + 1;
end

lenS = cellfun('length',w_S);                        % codelength
lenS_a = lenS .* P;                                
fprintf('The average codelength with Shannon Coding : %d\n', sum(lenS_a));   % average codelength

fprintf('\nProgram paused. Press enter to continue.\n\n');
pause;

%%  Coding 'Steve_Jobs_Speech' with Huffman Code

for i = 1:num_char                                   % whole characteristic in document
    char_doc{i} = char(stat(i));
end

fid = fopen('Coding_Huffman.txt','wt');              % Coding document
for i = 1:length
    [bool,inx] = ismember(str(i), char_doc);
    fprintf(fid, '%s', w_H{inx});
end
fclose(fid);
fprintf('\nHuffman Coding finished, See in ''Coding_Huffman.txt''. Press enter to continue.\n\n');
pause;

fids=fopen('Coding_Huffman.txt','r');                 % get code 
[A,COUNT]=fscanf(fids,'%c');

fidss = fopen('Decoding_Huffman.txt','wt');           % Decoding Huffman code
now = 1;
next = 1;
while (next <= COUNT)
    idx = 0;
    for i = 1:num_char
        if(strcmp(A(now:next), w_H{i}) == 1)          % compare to the codewords
            idx = i;
        end
    end
    if(idx ~= 0)
        fprintf(fidss, '%s', char_doc{idx});          % translate to symbols
        now = next + 1;
        next = now + 1;
    else
        next = next + 1;                              % + 1 
    end
end
fclose(fids);

fprintf('\nHuffman Decoding finished, See in ''Decoding_Huffman.txt''. Press enter to continue.\n\n');
pause;

%%  Coding 'Steve_Jobs_Speech' with Shannon Code

fid = fopen('Coding_Shannon.txt','wt');              % Coding document
for i = 1:length
    [bool,inx] = ismember(str(i), char_doc);
    fprintf(fid, '%s', w_S{inx});
end
fclose(fid);
fprintf('\nShannon Coding finished, See in ''Coding_Huffman.txt''. Press enter to continue.\n\n');
pause;

fids = fopen('Coding_Shannon.txt','r');               % get code 
[A,COUNT] = fscanf(fids,'%c');

fidss = fopen('Decoding_Shannon.txt','wt');           % Decoding Shannon code
now = 1;
next = 1;
while (next <= COUNT)
    idx = 0;
    for i = 1:num_char
        if(strcmp(A(now:next), w_S{i}) == 1)          % compare to the codewords
            idx = i;
        end
    end
    if(idx ~= 0)
        fprintf(fidss, '%s', char_doc{idx});          % translate to symbols
        now = next + 1;
        next = now + 1;
    else
        next = next + 1;                              % + 1
    end
end
fclose(fids);

fprintf('\nShannon Decoding finished, See in ''Decoding_Shannon.txt''. Press enter to finish this program.\n\n');
pause;