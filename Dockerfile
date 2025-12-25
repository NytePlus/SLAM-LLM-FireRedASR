FROM hub.szaic.com/hpc/ai_asr-jingpeng-ps-slm:v2.0

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

RUN pip install whisper_normalizer transformers==4.56.0