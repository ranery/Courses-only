function bin = dec_to_bin (m, N)
%
% Function : convert decimal to binary
% input    : m   --- decimal
%            N   --- the number of bits
% output   : bin --- binary
%
format long;
if (m >1) | (N == 0)                  % check input
    disp('error!');
    return;
end
count = 0;
temp_num = m;
record = zeros(1,N);
while(N)
    count = count + 1;               
    if (count > N)                    % codelength less than N
        N = 0;                        % stop loop
        break;
    end
    temp_num = temp_num * 2;          % convert fraction to binary 
    if (temp_num > 1)
        record(count) = 1;
        temp_num = temp_num-1;
    elseif (temp_num == 1)
        record(count) = 1;
        N = 0;                        % stop loop
    else
       record(count) = 0;    
    end
end
bin = record;