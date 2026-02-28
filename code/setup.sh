mkdir -p data/writingPrompts
wget https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz
curl -O https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz
tar -xzf writingPrompts.tar.gz \
    --strip-components=1 \
    -C data/writingPrompts \
    writingPrompts


SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RUPTURES=0.0.0 pip install -e .
