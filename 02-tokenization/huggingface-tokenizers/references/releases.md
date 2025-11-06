# Releases

Version history for this repository (99 releases).

## v0.22.1: v0.22.1
**Published:** 2025-09-19

# Release v0.22.1

Main change:
- Bump huggingface_hub upper version (#1866) from @Wauplin 
- chore(trainer): add and improve trainer signature (#1838) from @shenxiangzhuang 
- Some doc updates: c91d76ae558ca2dc1aa725959e65dc21bf1fed7e, 7b0217894c1e2baed7354ab41503841b47af7cf9, 57eb8d7d9564621221784f7949b9efdeb7a49ac1 



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.22.1)

---

## v0.22.0: v0.22.0
**Published:** 2025-08-29

## What's Changed
* Bump on-headers and compression in /tokenizers/examples/unstable_wasm/www by @dependabot[bot] in https://github.com/huggingface/tokenizers/pull/1827
* Implement `from_bytes` and `read_bytes` Methods in WordPiece Tokenizer for WebAssembly Compatibility by @sondalex in https://github.com/huggingface/tokenizers/pull/1758
* fix: use AHashMap to fix compile error by @b00f in https://github.com/huggingface/tokenizers/pull/1840
* New stream by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1856
* [docs] Add more decoders by @pcuenca in https://github.com/huggingface/tokenizers/pull/1849
* Fix missing parenthesis in `EncodingVisualizer.calculate_label_colors` by @Liam-DeVoe in https://github.com/huggingface/tokenizers/pull/1853
* Update quicktour.mdx re: Issue #1625 by @WilliamPLaCroix in https://github.com/huggingface/tokenizers/pull/1846
* remove stray comment by @sanderland in https://github.com/huggingface/tokenizers/pull/1831
* Fix typo in README by @aisk in https://github.com/huggingface/tokenizers/pull/1808
* RUSTSEC-2024-0436 - replace paste with pastey by @nystromjd in https://github.com/huggingface/tokenizers/pull/1834
* Tokenizer: Add native async bindings, via py03-async-runtimes. by @michaelfeil in https://github.com/huggingface/tokenizers/pull/1843

## New Contributors
* @b00f made their first contribution in https://github.com/huggingface/tokenizers/pull/1840
* @pcuenca made their first contribution in https://github.com/huggingface/tokenizers/pull/1849
* @Liam-DeVoe made their first contribution in https://github.com/huggingface/tokenizers/pull/1853
* @WilliamPLaCroix made their first contribution in https://github.com/huggingface/tokenizers/pull/1846
* @sanderland made their first contribution in https://github.com/huggingface/tokenizers/pull/1831
* @aisk made their first contribution in https://github.com/huggingface/tokenizers/pull/1808
* @nystromjd made their first contribution in https://github.com/huggingface/tokenizers/pull/1834
* @michaelfeil made their first contribution in https://github.com/huggingface/tokenizers/pull/1843

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.21.3...v0.22.0rc0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.22.0)

---

## v0.21.4: v0.21.4
**Published:** 2025-07-28

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.21.3...v0.21.4


No change, the 0.21.3 release failed, this is just a re-release.

https://github.com/huggingface/tokenizers/releases/tag/v0.21.3

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.21.4)

---

## v0.21.3: v0.21.3
**Published:** 2025-07-04

## What's Changed
* Clippy fixes. by @Narsil in https://github.com/huggingface/tokenizers/pull/1818
* Fixed an introduced backward breaking change in our Rust APIs.


**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.21.2...v0.21.3

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.21.3)

---

## v0.21.2: v0.21.2
**Published:** 2025-06-24

## What's Changed

This release if focused around some performance optimization, enabling broader python no gil support, and fixing some onig issues! 


* Update the release builds following 0.21.1. by @Narsil in https://github.com/huggingface/tokenizers/pull/1746
* replace lazy_static with stabilized std::sync::LazyLock in 1.80 by @sftse in https://github.com/huggingface/tokenizers/pull/1739
* Fix no-onig no-wasm builds by @414owen in https://github.com/huggingface/tokenizers/pull/1772
* Fix typos in strings and comments by @co63oc in https://github.com/huggingface/tokenizers/pull/1770
* Fix type notation of merges in BPE Python binding by @Coqueue in https://github.com/huggingface/tokenizers/pull/1766
* Bump http-proxy-middleware from 2.0.6 to 2.0.9 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1762
* Fix data path in test_continuing_prefix_trainer_mismatch by @GaetanLepage in https://github.com/huggingface/tokenizers/pull/1747
* clippy by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1781
* Update pyo3 and rust-numpy depends for no-gil/free-threading compat by @Qubitium in https://github.com/huggingface/tokenizers/pull/1774
* Use ApiBuilder::from_env() in from_pretrained function by @BenLocal in https://github.com/huggingface/tokenizers/pull/1737
* Upgrade onig, to get it compiling with GCC 15 by @414owen in https://github.com/huggingface/tokenizers/pull/1771
* Itertools upgrade by @sftse in https://github.com/huggingface/tokenizers/pull/1756
* Bump webpack-dev-server from 4.10.0 to 5.2.1 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1792
* Bump brace-expansion from 1.1.11 to 1.1.12 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1796
* Fix features blending into a paragraph by @bionicles in https://github.com/huggingface/tokenizers/pull/1798
* Adding throughput to benches to have a more consistent measure across by @Narsil in https://github.com/huggingface/tokenizers/pull/1800
* Upgrading dependencies. by @Narsil in https://github.com/huggingface/tokenizers/pull/1801
* [docs] Whitespace by @stevhliu in https://github.com/huggingface/tokenizers/pull/1785
* Hotfixing the stub. by @Narsil in https://github.com/huggingface/tokenizers/pull/1802
* Bpe clones by @sftse in https://github.com/huggingface/tokenizers/pull/1707
* Fixed Length Pre-Tokenizer by @jonvet in https://github.com/huggingface/tokenizers/pull/1713
* Consolidated optimization ahash dary compact str by @Narsil in https://github.com/huggingface/tokenizers/pull/1799
* 🚨 breaking: Fix training with special tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1617

## New Contributors
* @414owen made their first contribution in https://github.com/huggingface/tokenizers/pull/1772
* @co63oc made their first contribution in https://github.com/huggingface/tokenizers/pull/1770
* @Coqueue made their first contribution in https://github.com/huggingface/tokenizers/pull/1766
* @GaetanLepage made their first contribution in https://github.com/huggingface/tokenizers/pull/1747
* @Qubitium made their first contribution in https://github.com/huggingface/tokenizers/pull/1774
* @BenLocal made their first contribution in https://github.com/huggingface/tokenizers/pull/1737
* @bionicles made their first contribution in https://github.com/huggingface/tokenizers/pull/1798
* @stevhliu made their first contribution in https://github.com/huggingface/tokenizers/pull/1785
* @jonvet made their first contribution in https://github.com/huggingface/tokenizers/pull/1713

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.21.1...v0.21.2rc0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.21.2)

---

## v0.21.1: v0.21.1
**Published:** 2025-03-13

## What's Changed
* Update dev version and pyproject.toml by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1693
* Add feature flag hint to README.md, fixes #1633 by @sftse in https://github.com/huggingface/tokenizers/pull/1709
* Upgrade to PyO3 0.23 by @Narsil in https://github.com/huggingface/tokenizers/pull/1708
* Fixing the README. by @Narsil in https://github.com/huggingface/tokenizers/pull/1714
* Fix typo in Split docstrings by @Dylan-Harden3 in https://github.com/huggingface/tokenizers/pull/1701
* Fix typos by @tinyboxvk in https://github.com/huggingface/tokenizers/pull/1715
* Update documentation of Rust feature by @sondalex in https://github.com/huggingface/tokenizers/pull/1711
* Fix panic in DecodeStream::step due to incorrect index usage by @n0gu-furiosa in https://github.com/huggingface/tokenizers/pull/1699
* Fixing the stream by removing the read_index altogether. by @Narsil in https://github.com/huggingface/tokenizers/pull/1716
* Fixing NormalizedString append when normalized is empty. by @Narsil in https://github.com/huggingface/tokenizers/pull/1717
* 🚨 Support updating template processors by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1652.  Removed in this release to keep backware compatibility temporarily.
* Update metadata as Python3.7 and Python3.8 support was dropped by @earlytobed in https://github.com/huggingface/tokenizers/pull/1724
* Add rustls-tls feature by @torymur in https://github.com/huggingface/tokenizers/pull/1732

## New Contributors
* @Dylan-Harden3 made their first contribution in https://github.com/huggingface/tokenizers/pull/1701
* @sondalex made their first contribution in https://github.com/huggingface/tokenizers/pull/1711
* @n0gu-furiosa made their first contribution in https://github.com/huggingface/tokenizers/pull/1699
* @earlytobed made their first contribution in https://github.com/huggingface/tokenizers/pull/1724
* @torymur made their first contribution in https://github.com/huggingface/tokenizers/pull/1732

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.21.0...v0.21.1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.21.1)

---

## v0.21.1rc0: v0.21.1rc0
**Published:** 2025-03-12
**Pre-release**

## What's Changed
* Update dev version and pyproject.toml by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1693
* Add feature flag hint to README.md, fixes #1633 by @sftse in https://github.com/huggingface/tokenizers/pull/1709
* Upgrade to PyO3 0.23 by @Narsil in https://github.com/huggingface/tokenizers/pull/1708
* Fixing the README. by @Narsil in https://github.com/huggingface/tokenizers/pull/1714
* Fix typo in Split docstrings by @Dylan-Harden3 in https://github.com/huggingface/tokenizers/pull/1701
* Fix typos by @tinyboxvk in https://github.com/huggingface/tokenizers/pull/1715
* Update documentation of Rust feature by @sondalex in https://github.com/huggingface/tokenizers/pull/1711
* Fix panic in DecodeStream::step due to incorrect index usage by @n0gu-furiosa in https://github.com/huggingface/tokenizers/pull/1699
* Fixing the stream by removing the read_index altogether. by @Narsil in https://github.com/huggingface/tokenizers/pull/1716
* Fixing NormalizedString append when normalized is empty. by @Narsil in https://github.com/huggingface/tokenizers/pull/1717
* 🚨 Support updating template processors by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1652
* Update metadata as Python3.7 and Python3.8 support was dropped by @earlytobed in https://github.com/huggingface/tokenizers/pull/1724
* Add rustls-tls feature by @torymur in https://github.com/huggingface/tokenizers/pull/1732

## New Contributors
* @Dylan-Harden3 made their first contribution in https://github.com/huggingface/tokenizers/pull/1701
* @sondalex made their first contribution in https://github.com/huggingface/tokenizers/pull/1711
* @n0gu-furiosa made their first contribution in https://github.com/huggingface/tokenizers/pull/1699
* @earlytobed made their first contribution in https://github.com/huggingface/tokenizers/pull/1724
* @torymur made their first contribution in https://github.com/huggingface/tokenizers/pull/1732

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.21.0...v0.21.1rc0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.21.1rc0)

---

## v0.21.0: Release v0.21.0 
**Published:** 2024-11-15

## Release ~v0.20.4~ v0.21.0 
* More cache options. by @Narsil in https://github.com/huggingface/tokenizers/pull/1675
* Disable caching for long strings. by @Narsil in https://github.com/huggingface/tokenizers/pull/1676
* Testing ABI3 wheels to reduce number of wheels by @Narsil in https://github.com/huggingface/tokenizers/pull/1674
* Adding an API for decode streaming. by @Narsil in https://github.com/huggingface/tokenizers/pull/1677
* Decode stream python by @Narsil in https://github.com/huggingface/tokenizers/pull/1678
* Fix encode_batch and encode_batch_fast to accept ndarrays again by @diliop in  https://github.com/huggingface/tokenizers/pull/1679 

We also no longer support python 3.7 or 3.8 (similar to transformers) as they are deprecated.

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.20.3...v0.21.0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.21.0)

---

## v0.20.3: v0.20.3
**Published:** 2024-11-05

## What's Changed
There was a breaking change in `0.20.3` for tuple inputs of `encode_batch`! 
* fix pylist by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1673
* [MINOR:TYPO] Fix docstrings by @cakiki in https://github.com/huggingface/tokenizers/pull/1653

## New Contributors
* @cakiki made their first contribution in https://github.com/huggingface/tokenizers/pull/1653

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.20.2...v0.20.3

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.20.3)

---

## v0.20.2: v0.20.2
**Published:** 2024-11-04

# Release v0.20.2

Thanks a MILE to @diliop we now have support for python 3.13! 🥳 

## What's Changed
* Bump cookie and express in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1648
* Fix off-by-one error in tokenizer::normalizer::Range::len by @rlanday in https://github.com/huggingface/tokenizers/pull/1638
* Arg name correction: auth_token -> token by @rravenel in https://github.com/huggingface/tokenizers/pull/1621
* Unsound call of `set_var` by @sftse in https://github.com/huggingface/tokenizers/pull/1664
* Add safety comments by @Manishearth in https://github.com/huggingface/tokenizers/pull/1651
* Bump actions/checkout to v4 by @tinyboxvk in https://github.com/huggingface/tokenizers/pull/1667
* PyO3 0.22 by @diliop in https://github.com/huggingface/tokenizers/pull/1665
* Bump actions versions by @tinyboxvk in https://github.com/huggingface/tokenizers/pull/1669

## New Contributors
* @rlanday made their first contribution in https://github.com/huggingface/tokenizers/pull/1638
* @rravenel made their first contribution in https://github.com/huggingface/tokenizers/pull/1621
* @sftse made their first contribution in https://github.com/huggingface/tokenizers/pull/1664
* @Manishearth made their first contribution in https://github.com/huggingface/tokenizers/pull/1651
* @tinyboxvk made their first contribution in https://github.com/huggingface/tokenizers/pull/1667
* @diliop made their first contribution in https://github.com/huggingface/tokenizers/pull/1665

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.20.1...v0.20.2

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.20.2)

---

## v0.20.1: Release v0.20.1
**Published:** 2024-10-10

## What's Changed
The most awaited `offset` issue with `Llama` is fixed 🥳 

* Update README.md by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1608
* fix benchmark file link by @152334H in https://github.com/huggingface/tokenizers/pull/1610
* Bump actions/download-artifact from 3 to 4.1.7 in /.github/workflows by @dependabot in https://github.com/huggingface/tokenizers/pull/1626
* [`ignore_merges`] Fix offsets by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1640
* Bump body-parser and express in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1629
* Bump serve-static and express in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1630
* Bump send and express in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1631
* Bump webpack from 5.76.0 to 5.95.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1641
* Fix documentation build by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1642
* style: simplify string formatting for readability by @hamirmahal in https://github.com/huggingface/tokenizers/pull/1632

## New Contributors
* @152334H made their first contribution in https://github.com/huggingface/tokenizers/pull/1610
* @hamirmahal made their first contribution in https://github.com/huggingface/tokenizers/pull/1632

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.20.0...v0.20.1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.20.1)

---

## v0.20.0: Release v0.20.0: faster encode, better python support
**Published:** 2024-08-08

# Release v0.20.0

This release is focused on **performances** and **user experience**. 

## Performances:
First off, we did a bit of benchmarking, and found some place for improvement for us!
With a few minor changes (mostly #1587) here is what we get on `Llama3` running on a g6 instances on AWS `https://github.com/huggingface/tokenizers/blob/main/bindings/python/benches/test_tiktoken.py` : 
![image](https://github.com/user-attachments/assets/e6838866-ec76-44ce-a7b6-532e56971234)

## Python API
We shipped better deserialization errors in general, and support for `__str__` and `__repr__` for all the object. This allows for a lot easier debugging see this:
```python3
>>> from tokenizers import Tokenizer;
>>> tokenizer = Tokenizer.from_pretrained("bert-base-uncased");
>>> print(tokenizer)
Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[{"id":0, "content":"[PAD]", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":100, "content":"[UNK]", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":101, "content":"[CLS]", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":102, "content":"[SEP]", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":103, "content":"[MASK]", "single_word":False, "lstrip":False, "rstrip":False, ...}], normalizer=BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True), pre_tokenizer=BertPreTokenizer(), post_processor=TemplateProcessing(single=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0)], pair=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0), Sequence(id=B, type_id=1), SpecialToken(id="[SEP]", type_id=1)], special_tokens={"[CLS]":SpecialToken(id="[CLS]", ids=[101], tokens=["[CLS]"]), "[SEP]":SpecialToken(id="[SEP]", ids=[102], tokens=["[SEP]"])}), decoder=WordPiece(prefix="##", cleanup=True), model=WordPiece(unk_token="[UNK]", continuing_subword_prefix="##", max_input_chars_per_word=100, vocab={"[PAD]":0, "[unused0]":1, "[unused1]":2, "[unused2]":3, "[unused3]":4, ...}))

>>> tokenizer
Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[{"id":0, "content":"[PAD]", "single_word":False, "lstrip":False, "rstrip":False, "normalized":False, "special":True}, {"id":100, "content":"[UNK]", "single_word":False, "lstrip":False, "rstrip":False, "normalized":False, "special":True}, {"id":101, "content":"[CLS]", "single_word":False, "lstrip":False, "rstrip":False, "normalized":False, "special":True}, {"id":102, "content":"[SEP]", "single_word":False, "lstrip":False, "rstrip":False, "normalized":False, "special":True}, {"id":103, "content":"[MASK]", "single_word":False, "lstrip":False, "rstrip":False, "normalized":False, "special":True}], normalizer=BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True), pre_tokenizer=BertPreTokenizer(), post_processor=TemplateProcessing(single=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0)], pair=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0), Sequence(id=B, type_id=1), SpecialToken(id="[SEP]", type_id=1)], special_tokens={"[CLS]":SpecialToken(id="[CLS]", ids=[101], tokens=["[CLS]"]), "[SEP]":SpecialToken(id="[SEP]", ids=[102], tokens=["[SEP]"])}), decoder=WordPiece(prefix="##", cleanup=True), model=WordPiece(unk_token="[UNK]", continuing_subword_prefix="##", max_input_chars_per_word=100, vocab={"[PAD]":0, "[unused0]":1, "[unused1]":2, ...}))
```

The `pre_tokenizer.Sequence` and `normalizer.Sequence` are also more accessible now:
```python 
from tokenizers import normalizers
norm = normalizers.Sequence([normalizers.Strip(), normalizers.BertNormalizer()])
norm[0]
norm[1].lowercase=False
```

## What's Changed
* remove enforcement of non special when adding tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1521
* [BREAKING CHANGE] Ignore added_tokens (both special and not) in the decoder by @Narsil in https://github.com/huggingface/tokenizers/pull/1513
* Make `USED_PARALLELISM` atomic by @nathaniel-daniel in https://github.com/huggingface/tokenizers/pull/1532
* Fixing for clippy 1.78 by @Narsil in https://github.com/huggingface/tokenizers/pull/1548
* feat(ci): add trufflehog secrets detection by @McPatate in https://github.com/huggingface/tokenizers/pull/1551
* Switch from `cached_download` to `hf_hub_download` in tests by @Wauplin in https://github.com/huggingface/tokenizers/pull/1547
* Fix "dictionnary" typo by @nprisbrey in https://github.com/huggingface/tokenizers/pull/1511
* make sure we don't warn on empty tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1554
* Enable `dropout = 0.0` as an equivalent to `none` in BPE by @mcognetta in https://github.com/huggingface/tokenizers/pull/1550
* Revert "[BREAKING CHANGE] Ignore added_tokens (both special and not) … by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1569
* Add bytelevel normalizer to fix decode when adding tokens to BPE by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1555
* Fix clippy + feature test management. by @Narsil in https://github.com/huggingface/tokenizers/pull/1580
* Bump spm_precompiled to 0.1.3 by @MikeIvanichev in https://github.com/huggingface/tokenizers/pull/1571
* Add benchmark vs tiktoken by @Narsil in https://github.com/huggingface/tokenizers/pull/1582
* Fixing the benchmark. by @Narsil in https://github.com/huggingface/tokenizers/pull/1583
* Tiny improvement by @Narsil in https://github.com/huggingface/tokenizers/pull/1585
* Enable fancy regex by @Narsil in https://github.com/huggingface/tokenizers/pull/1586
* Fixing release CI strict (taken from safetensors). by @Narsil in https://github.com/huggingface/tokenizers/pull/1593
* Adding some serialization testing around the wrapper. by @Narsil in https://github.com/huggingface/tokenizers/pull/1594
* Add-legacy-tests by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1597
* Adding a few tests for decoder deserialization. by @Narsil in https://github.com/huggingface/tokenizers/pull/1598
* Better serialization error by @Narsil in https://github.com/huggingface/tokenizers/pull/1595
* Add test normalizers by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1600
* Improve decoder deserialization by @Narsil in https://github.com/huggingface/tokenizers/pull/1599
* Using serde (serde_pyo3) to get __str__ and __repr__ easily. by @Narsil in https://github.com/huggingface/tokenizers/pull/1588
* Merges cannot handle tokens containing spaces. by @Narsil in https://github.com/huggingface/tokenizers/pull/909
* Fix doc about split by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1591
* Support `None` to reset pre_tokenizers and normalizers, and index sequences by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1590
* Fix strip python type by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1602
* Tests + Deserialization improvement for normalizers. by @Narsil in https://github.com/huggingface/tokenizers/pull/1604
* add deserialize for pre tokenizers by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1603
* Perf improvement 16% by removing offsets. by @Narsil in https://github.com/huggingface/tokenizers/pull/1587

## New Contributors
* @nathaniel-daniel made their first contribution in https://github.com/huggingface/tokenizers/pull/1532
* @nprisbrey made their first contribution in https://github.com/huggingface/tokenizers/pull/1511
* @mcognetta made their first contribution in https://github.com/huggingface/tokenizers/pull/1550
* @MikeIvanichev made their first contribution in https://github.com/huggingface/tokenizers/pull/1571

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.19.1...v0.20.0rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.20.0)

---

## v0.19.1: v0.19.1
**Published:** 2024-04-17

## What's Changed
* add serialization for `ignore_merges` by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1504


**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.19.0...v0.19.1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.19.1)

---

## v0.19.0: v0.19.0
**Published:** 2024-04-17

## What's Changed
* chore: Remove CLI - this was originally intended for local development by @bryantbiggs in https://github.com/huggingface/tokenizers/pull/1442
* [`remove black`] And use ruff by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1436
* Bump ip from 2.0.0 to 2.0.1 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1456
* Added ability to inspect a 'Sequence' decoder and the `AddedVocabulary`. by @eaplatanios in https://github.com/huggingface/tokenizers/pull/1443
* 🚨🚨 BREAKING CHANGE 🚨🚨: (add_prefix_space dropped everything is using prepend_scheme enum instead) Refactor metaspace by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1476
* Add more support for tiktoken based tokenizers by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1493
* PyO3 0.21. by @Narsil in https://github.com/huggingface/tokenizers/pull/1494
* Remove 3.13 (potential undefined behavior.) by @Narsil in https://github.com/huggingface/tokenizers/pull/1497
* Bumping all versions 3 times (ty transformers :) ) by @Narsil in https://github.com/huggingface/tokenizers/pull/1498
* Fixing doc. by @Narsil in https://github.com/huggingface/tokenizers/pull/1499


**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.15.2...v0.19.0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.19.0)

---

## v0.19.0rc0: v0.19.0rc0
**Published:** 2024-04-16
**Pre-release**

Bumping 3 versions because of this: https://github.com/huggingface/transformers/blob/60dea593edd0b94ee15dc3917900b26e3acfbbee/setup.py#L177

## What's Changed
* chore: Remove CLI - this was originally intended for local development by @bryantbiggs in https://github.com/huggingface/tokenizers/pull/1442
* [`remove black`] And use ruff by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1436
* Bump ip from 2.0.0 to 2.0.1 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1456
* Added ability to inspect a 'Sequence' decoder and the `AddedVocabulary`. by @eaplatanios in https://github.com/huggingface/tokenizers/pull/1443
* 🚨🚨 BREAKING CHANGE 🚨🚨:  (add_prefix_space dropped everything is using prepend_scheme enum instead) Refactor metaspace by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1476
* Add more support for tiktoken based tokenizers by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1493
* PyO3 0.21. by @Narsil in https://github.com/huggingface/tokenizers/pull/1494
* Remove 3.13 (potential undefined behavior.) by @Narsil in https://github.com/huggingface/tokenizers/pull/1497
* Bumping all versions 3 times (ty transformers :) ) by @Narsil in https://github.com/huggingface/tokenizers/pull/1498


**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.15.2...v0.19.0rc0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.19.0rc0)

---

## v0.15.2: v0.15.2
**Published:** 2024-02-12

## What's Changed
Big shoutout to @rlrs for [the fast replace normalizers](https://github.com/huggingface/tokenizers/pull/1413) PR. This boosts the performances of the tokenizers: 
![image](https://github.com/huggingface/tokenizers/assets/48595927/d8ee81b1-6d92-43d4-b74c-8775727763e3)

* chore: Update dependencies to latest supported versions by @bryantbiggs in https://github.com/huggingface/tokenizers/pull/1441
* Convert word counts to u64 by @stephenroller in https://github.com/huggingface/tokenizers/pull/1433
* Efficient Replace normalizer by @rlrs in https://github.com/huggingface/tokenizers/pull/1413

## New Contributors
* @bryantbiggs made their first contribution in https://github.com/huggingface/tokenizers/pull/1441
* @stephenroller made their first contribution in https://github.com/huggingface/tokenizers/pull/1433
* @rlrs made their first contribution in https://github.com/huggingface/tokenizers/pull/1413

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.15.1...v0.15.2rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.15.2)

---

## v0.15.1: v0.15.1
**Published:** 2024-01-22

## What's Changed
* udpate to version = "0.15.1-dev0" by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1390
* Derive `Clone` on `Tokenizer`, add `Encoding.into_tokens()` method by @epwalsh in https://github.com/huggingface/tokenizers/pull/1381
* Stale bot. by @Narsil in https://github.com/huggingface/tokenizers/pull/1404
* Fix doc links in readme by @Pierrci in https://github.com/huggingface/tokenizers/pull/1367
* Faster HF dataset iteration in docs by @mariosasko in https://github.com/huggingface/tokenizers/pull/1414
* Add quick doc to byte_level.rs by @steventrouble in https://github.com/huggingface/tokenizers/pull/1420
* Fix make bench. by @Narsil in https://github.com/huggingface/tokenizers/pull/1428
* Bump follow-redirects from 1.15.1 to 1.15.4 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1430
* pyo3: update to 0.20 by @mikelui in https://github.com/huggingface/tokenizers/pull/1386
* Encode special tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1437
* Update release for python3.12 windows by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1438

## New Contributors
* @steventrouble made their first contribution in https://github.com/huggingface/tokenizers/pull/1420

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.15.0...v0.15.1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.15.1)

---

## v0.15.1.rc0: v0.15.1.rc0
**Published:** 2024-01-18
**Pre-release**

## What's Changed
* pyo3: update to 0.19 by @mikelui in https://github.com/huggingface/tokenizers/pull/1322
* Add `expect()` for disabling truncation by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1316
* Re-using scritpts from safetensors. by @Narsil in https://github.com/huggingface/tokenizers/pull/1328
* Reduce number of different revisions by 1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1329
* Python 38 arm by @Narsil in https://github.com/huggingface/tokenizers/pull/1330
* Move to maturing mimicking move for `safetensors`. + Rewritten node bindings. by @Narsil in https://github.com/huggingface/tokenizers/pull/1331
* Updating the docs with the new command. by @Narsil in https://github.com/huggingface/tokenizers/pull/1333
* Update added tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1335
* update package version for dev by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1339
* Added ability to inspect a 'Sequence' pre-tokenizer. by @eaplatanios in https://github.com/huggingface/tokenizers/pull/1341
* Let's allow hf_hub < 1.0 by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1344
* Fixing the progressbar. by @Narsil in https://github.com/huggingface/tokenizers/pull/1353
* Preparing release. by @Narsil in https://github.com/huggingface/tokenizers/pull/1355
* fix a clerical error  in the comment by @tiandiweizun in https://github.com/huggingface/tokenizers/pull/1356
* fix: remove useless token by @rtrompier in https://github.com/huggingface/tokenizers/pull/1371
* Bump @babel/traverse from 7.22.11 to 7.23.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1370
* Allow hf_hub 0.18 by @mariosasko in https://github.com/huggingface/tokenizers/pull/1383
* Allow `huggingface_hub<1.0` by @Wauplin in https://github.com/huggingface/tokenizers/pull/1385
* [`pre_tokenizers`] Fix sentencepiece based Metaspace by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1357
* udpate to version = "0.15.1-dev0" by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1390
* Derive `Clone` on `Tokenizer`, add `Encoding.into_tokens()` method by @epwalsh in https://github.com/huggingface/tokenizers/pull/1381
* Stale bot. by @Narsil in https://github.com/huggingface/tokenizers/pull/1404
* Fix doc links in readme by @Pierrci in https://github.com/huggingface/tokenizers/pull/1367
* Faster HF dataset iteration in docs by @mariosasko in https://github.com/huggingface/tokenizers/pull/1414
* Add quick doc to byte_level.rs by @steventrouble in https://github.com/huggingface/tokenizers/pull/1420
* Fix make bench. by @Narsil in https://github.com/huggingface/tokenizers/pull/1428
* Bump follow-redirects from 1.15.1 to 1.15.4 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1430
* pyo3: update to 0.20 by @mikelui in https://github.com/huggingface/tokenizers/pull/1386

## New Contributors
* @mikelui made their first contribution in https://github.com/huggingface/tokenizers/pull/1322
* @eaplatanios made their first contribution in https://github.com/huggingface/tokenizers/pull/1341
* @tiandiweizun made their first contribution in https://github.com/huggingface/tokenizers/pull/1356
* @rtrompier made their first contribution in https://github.com/huggingface/tokenizers/pull/1371
* @mariosasko made their first contribution in https://github.com/huggingface/tokenizers/pull/1383
* @Wauplin made their first contribution in https://github.com/huggingface/tokenizers/pull/1385
* @steventrouble made their first contribution in https://github.com/huggingface/tokenizers/pull/1420

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.4.rc2...v0.15.1.rc0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.15.1.rc0)

---

## v0.15.0: 
**Published:** 2023-11-14

## What's Changed
* fix a clerical error  in the comment by @tiandiweizun in https://github.com/huggingface/tokenizers/pull/1356
* fix: remove useless token by @rtrompier in https://github.com/huggingface/tokenizers/pull/1371
* Bump @babel/traverse from 7.22.11 to 7.23.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1370
* Allow hf_hub 0.18 by @mariosasko in https://github.com/huggingface/tokenizers/pull/1383
* Allow `huggingface_hub<1.0` by @Wauplin in https://github.com/huggingface/tokenizers/pull/1385
* [`pre_tokenizers`] Fix sentencepiece based Metaspace by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1357

## New Contributors
* @tiandiweizun made their first contribution in https://github.com/huggingface/tokenizers/pull/1356
* @rtrompier made their first contribution in https://github.com/huggingface/tokenizers/pull/1371
* @mariosasko made their first contribution in https://github.com/huggingface/tokenizers/pull/1383
* @Wauplin made their first contribution in https://github.com/huggingface/tokenizers/pull/1385

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.14.1...v0.15.0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.15.0)

---

## v0.14.1: v0.14.1
**Published:** 2023-10-06

## What's Changed
* Fix conda release by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1211
* Fix node release by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1212
* Printing warning to stderr. by @Narsil in https://github.com/huggingface/tokenizers/pull/1222
* Fixing padding_left sequence_ids. by @Narsil in https://github.com/huggingface/tokenizers/pull/1233
* Use LTO for release and benchmark builds by @csko in https://github.com/huggingface/tokenizers/pull/1157
* fix unigram.rs test_sample() by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1244
* implement a simple max_sentencepiece_length into BPE by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1228
* Makes `decode` and `decode_batch` work on borrowed content. by @mfuntowicz in https://github.com/huggingface/tokenizers/pull/1251
* Update all GH Actions with dependency on actions/checkout by @mfuntowicz in https://github.com/huggingface/tokenizers/pull/1256
* Parallelize unigram trainer by @mishig25 in https://github.com/huggingface/tokenizers/pull/976
* Update unigram/trainer.rs by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1257
* Fixing broken link. by @Narsil in https://github.com/huggingface/tokenizers/pull/1268
* fix documentation regarding regex  by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1264
* Update Cargo.toml by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1266
* Update README.md - Broken link by @sbhavani in https://github.com/huggingface/tokenizers/pull/1272
* [doc build] Use secrets by @mishig25 in https://github.com/huggingface/tokenizers/pull/1273
* Improve error for truncation with too high stride by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1275
* Add unigram bytefallback by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1217
* revise type specification by @hiroshi-matsuda-rit in https://github.com/huggingface/tokenizers/pull/1289
* Bump tough-cookie from 4.0.0 to 4.1.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1291
* Update path name: master -> main by @bact in https://github.com/huggingface/tokenizers/pull/1292
* import Tuple from typing by @kellymarchisio in https://github.com/huggingface/tokenizers/pull/1295
* Fixing clippy warnings on 1.71. by @Narsil in https://github.com/huggingface/tokenizers/pull/1296
* Bump word-wrap from 1.2.3 to 1.2.4 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1299
* feat: Added CITATION.cff. by @SamuelLarkin in https://github.com/huggingface/tokenizers/pull/1302
* Single warning for holes. by @Narsil in https://github.com/huggingface/tokenizers/pull/1303
* Give error when initializing tokenizer with too high stride by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1306
* Handle when precompiled charsmap is empty by @kellymarchisio in https://github.com/huggingface/tokenizers/pull/1308
* Derive clone for TrainerWrapper by @jonatanklosko in https://github.com/huggingface/tokenizers/pull/1317
* CD backports by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1318
* 0.13.4.rc1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1319
* Release all at once for simplicity. by @Narsil in https://github.com/huggingface/tokenizers/pull/1320
* Fix stride condition. by @Narsil in https://github.com/huggingface/tokenizers/pull/1321
* pyo3: update to 0.19 by @mikelui in https://github.com/huggingface/tokenizers/pull/1322
* Add `expect()` for disabling truncation by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1316
* Re-using scritpts from safetensors. by @Narsil in https://github.com/huggingface/tokenizers/pull/1328
* Reduce number of different revisions by 1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1329
* Python 38 arm by @Narsil in https://github.com/huggingface/tokenizers/pull/1330
* Move to maturing mimicking move for `safetensors`. + Rewritten node bindings. by @Narsil in https://github.com/huggingface/tokenizers/pull/1331
* Updating the docs with the new command. by @Narsil in https://github.com/huggingface/tokenizers/pull/1333
* Update added tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1335
* update package version for dev by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1339
* Added ability to inspect a 'Sequence' pre-tokenizer. by @eaplatanios in https://github.com/huggingface/tokenizers/pull/1341
* Let's allow hf_hub < 1.0 by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1344
* Fixing the progressbar. by @Narsil in https://github.com/huggingface/tokenizers/pull/1353
* Preparing release. by @Narsil in https://github.com/huggingface/tokenizers/pull/1355

## New Contributors
* @csko made their first contribution in https://github.com/huggingface/tokenizers/pull/1157
* @chris-ha458 made their first contribution in https://github.com/huggingface/tokenizers/pull/1244
* @sbhavani made their first contribution in https://github.com/huggingface/tokenizers/pull/1272
* @boyleconnor made their first contribution in https://github.com/huggingface/tokenizers/pull/1275
* @hiroshi-matsuda-rit made their first contribution in https://github.com/huggingface/tokenizers/pull/1289
* @bact made their first contribution in https://github.com/huggingface/tokenizers/pull/1292
* @kellymarchisio made their first contribution in https://github.com/huggingface/tokenizers/pull/1295
* @SamuelLarkin made their first contribution in https://github.com/huggingface/tokenizers/pull/1302
* @jonatanklosko made their first contribution in https://github.com/huggingface/tokenizers/pull/1317
* @mikelui made their first contribution in https://github.com/huggingface/tokenizers/pull/1322
* @eaplatanios made their first contribution in https://github.com/huggingface/tokenizers/pull/1341

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.3...v0.14.1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.14.1)

---

## v0.14.1rc1: v0.14.1rc1
**Published:** 2023-10-05
**Pre-release**

## What's Changed
* pyo3: update to 0.19 by @mikelui in https://github.com/huggingface/tokenizers/pull/1322
* Add `expect()` for disabling truncation by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1316
* Re-using scritpts from safetensors. by @Narsil in https://github.com/huggingface/tokenizers/pull/1328
* Reduce number of different revisions by 1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1329
* Python 38 arm by @Narsil in https://github.com/huggingface/tokenizers/pull/1330
* Move to maturing mimicking move for `safetensors`. + Rewritten node bindings. by @Narsil in https://github.com/huggingface/tokenizers/pull/1331
* Updating the docs with the new command. by @Narsil in https://github.com/huggingface/tokenizers/pull/1333
* Update added tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1335
* update package version for dev by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1339
* Added ability to inspect a 'Sequence' pre-tokenizer. by @eaplatanios in https://github.com/huggingface/tokenizers/pull/1341
* Let's allow hf_hub < 1.0 by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1344
* Fixing the progressbar. by @Narsil in https://github.com/huggingface/tokenizers/pull/1353

## New Contributors
* @mikelui made their first contribution in https://github.com/huggingface/tokenizers/pull/1322
* @eaplatanios made their first contribution in https://github.com/huggingface/tokenizers/pull/1341

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.4.rc2...v0.14.1rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.14.1rc1)

---

## v0.14.0: v0.14.0
**Published:** 2023-09-07

⚠️ Reworks the release pipeline. Other breaking changes ⚠️ :
- #1335, AddedToken is reworked, `is_special_token` rename to `special` for consistency
- feature http is now `OFF` by default, and depends on hf-hub instead of cached_path (updated cache directory, better sync implementation)
- Removed SSL link on the python package, calling huggingface_hub directly instead.
- New dependency : huggingface_hub  (while we deprecate Tokenizer.from_pretrained(...) to Tokenizer.from_file(hugginngface_hub.hf_hub_download(MODEL_ID, "tokenizer.json")


## What's Changed
* Fix conda release by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1211
* Fix node release by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1212
* Printing warning to stderr. by @Narsil in https://github.com/huggingface/tokenizers/pull/1222
* Fixing padding_left sequence_ids. by @Narsil in https://github.com/huggingface/tokenizers/pull/1233
* Use LTO for release and benchmark builds by @csko in https://github.com/huggingface/tokenizers/pull/1157
* fix unigram.rs test_sample() by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1244
* implement a simple max_sentencepiece_length into BPE by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1228
* Makes `decode` and `decode_batch` work on borrowed content. by @mfuntowicz in https://github.com/huggingface/tokenizers/pull/1251
* Update all GH Actions with dependency on actions/checkout by @mfuntowicz in https://github.com/huggingface/tokenizers/pull/1256
* Parallelize unigram trainer by @mishig25 in https://github.com/huggingface/tokenizers/pull/976
* Update unigram/trainer.rs by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1257
* Fixing broken link. by @Narsil in https://github.com/huggingface/tokenizers/pull/1268
* fix documentation regarding regex  by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1264
* Update Cargo.toml by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1266
* Update README.md - Broken link by @sbhavani in https://github.com/huggingface/tokenizers/pull/1272
* [doc build] Use secrets by @mishig25 in https://github.com/huggingface/tokenizers/pull/1273
* Improve error for truncation with too high stride by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1275
* Add unigram bytefallback by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1217
* revise type specification by @hiroshi-matsuda-rit in https://github.com/huggingface/tokenizers/pull/1289
* Bump tough-cookie from 4.0.0 to 4.1.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1291
* Update path name: master -> main by @bact in https://github.com/huggingface/tokenizers/pull/1292
* import Tuple from typing by @kellymarchisio in https://github.com/huggingface/tokenizers/pull/1295
* Fixing clippy warnings on 1.71. by @Narsil in https://github.com/huggingface/tokenizers/pull/1296
* Bump word-wrap from 1.2.3 to 1.2.4 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1299
* feat: Added CITATION.cff. by @SamuelLarkin in https://github.com/huggingface/tokenizers/pull/1302
* Single warning for holes. by @Narsil in https://github.com/huggingface/tokenizers/pull/1303
* Give error when initializing tokenizer with too high stride by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1306
* Handle when precompiled charsmap is empty by @kellymarchisio in https://github.com/huggingface/tokenizers/pull/1308
* Derive clone for TrainerWrapper by @jonatanklosko in https://github.com/huggingface/tokenizers/pull/1317
* CD backports by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1318
* 0.13.4.rc1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1319
* Release all at once for simplicity. by @Narsil in https://github.com/huggingface/tokenizers/pull/1320
* Fix stride condition. by @Narsil in https://github.com/huggingface/tokenizers/pull/1321
* pyo3: update to 0.19 by @mikelui in https://github.com/huggingface/tokenizers/pull/1322
* Add `expect()` for disabling truncation by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1316
* Re-using scritpts from safetensors. by @Narsil in https://github.com/huggingface/tokenizers/pull/1328
* Reduce number of different revisions by 1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1329
* Python 38 arm by @Narsil in https://github.com/huggingface/tokenizers/pull/1330
* Move to maturing mimicking move for `safetensors`. + Rewritten node bindings. by @Narsil in https://github.com/huggingface/tokenizers/pull/1331
* Updating the docs with the new command. by @Narsil in https://github.com/huggingface/tokenizers/pull/1333
* Update added tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1335

## New Contributors
* @csko made their first contribution in https://github.com/huggingface/tokenizers/pull/1157
* @chris-ha458 made their first contribution in https://github.com/huggingface/tokenizers/pull/1244
* @sbhavani made their first contribution in https://github.com/huggingface/tokenizers/pull/1272
* @boyleconnor made their first contribution in https://github.com/huggingface/tokenizers/pull/1275
* @hiroshi-matsuda-rit made their first contribution in https://github.com/huggingface/tokenizers/pull/1289
* @bact made their first contribution in https://github.com/huggingface/tokenizers/pull/1292
* @kellymarchisio made their first contribution in https://github.com/huggingface/tokenizers/pull/1295
* @SamuelLarkin made their first contribution in https://github.com/huggingface/tokenizers/pull/1302
* @jonatanklosko made their first contribution in https://github.com/huggingface/tokenizers/pull/1317
* @mikelui made their first contribution in https://github.com/huggingface/tokenizers/pull/1322

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.3...v0.14.0

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.14.0)

---

## v0.14.0.rc1: v0.14.0.rc1
**Published:** 2023-09-07
**Pre-release**

Reworks the release pipeline. Other breaking changes are mostly related to https://github.com/huggingface/tokenizers/pull/1335, where AddedToken is reworked

## What's Changed
* pyo3: update to 0.19 by @mikelui in https://github.com/huggingface/tokenizers/pull/1322
* Add `expect()` for disabling truncation by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1316
* Re-using scritpts from safetensors. by @Narsil in https://github.com/huggingface/tokenizers/pull/1328
* Reduce number of different revisions by 1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1329
* Python 38 arm by @Narsil in https://github.com/huggingface/tokenizers/pull/1330
* Move to maturing mimicking move for `safetensors`. + Rewritten node bindings. by @Narsil in https://github.com/huggingface/tokenizers/pull/1331
* Updating the docs with the new command. by @Narsil in https://github.com/huggingface/tokenizers/pull/1333
* Update added tokens by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1335

## New Contributors
* @mikelui made their first contribution in https://github.com/huggingface/tokenizers/pull/1322

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.4.rc2...v0.14.0.rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.14.0.rc1)

---

## v0.13.4.rc3: v0.13.4.rc3
**Published:** 2023-08-23

Mostly checking the new release scripts actually work.

## What's Changed
* pyo3: update to 0.19 by @mikelui in https://github.com/huggingface/tokenizers/pull/1322
* Add `expect()` for disabling truncation by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1316
* Re-using scritpts from safetensors. by @Narsil in https://github.com/huggingface/tokenizers/pull/1328

## New Contributors
* @mikelui made their first contribution in https://github.com/huggingface/tokenizers/pull/1322

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.4.rc2...v0.13.4.rc3

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.4.rc3)

---

## v0.13.4.rc2: v0.13.4.rc2
**Published:** 2023-08-14
**Pre-release**

## What's Changed
* Fix stride condition. by @Narsil in https://github.com/huggingface/tokenizers/pull/1321


**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.4.rc1...v0.13.4.rc2

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.4.rc2)

---

## v0.13.4.rc1: Python v0.13.4.rc1
**Published:** 2023-08-14
**Pre-release**

## What's Changed
* Update all GH Actions with dependency on actions/checkout by @mfuntowicz in https://github.com/huggingface/tokenizers/pull/1256
* Parallelize unigram trainer by @mishig25 in https://github.com/huggingface/tokenizers/pull/976
* Update unigram/trainer.rs by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1257
* Fixing broken link. by @Narsil in https://github.com/huggingface/tokenizers/pull/1268
* fix documentation regarding regex  by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1264
* Update Cargo.toml by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1266
* Update README.md - Broken link by @sbhavani in https://github.com/huggingface/tokenizers/pull/1272
* [doc build] Use secrets by @mishig25 in https://github.com/huggingface/tokenizers/pull/1273
* Improve error for truncation with too high stride by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1275
* Add unigram bytefallback by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1217
* revise type specification by @hiroshi-matsuda-rit in https://github.com/huggingface/tokenizers/pull/1289
* Bump tough-cookie from 4.0.0 to 4.1.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1291
* Update path name: master -> main by @bact in https://github.com/huggingface/tokenizers/pull/1292
* import Tuple from typing by @kellymarchisio in https://github.com/huggingface/tokenizers/pull/1295
* Fixing clippy warnings on 1.71. by @Narsil in https://github.com/huggingface/tokenizers/pull/1296
* Bump word-wrap from 1.2.3 to 1.2.4 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1299
* feat: Added CITATION.cff. by @SamuelLarkin in https://github.com/huggingface/tokenizers/pull/1302
* Single warning for holes. by @Narsil in https://github.com/huggingface/tokenizers/pull/1303
* Give error when initializing tokenizer with too high stride by @boyleconnor in https://github.com/huggingface/tokenizers/pull/1306
* Handle when precompiled charsmap is empty by @kellymarchisio in https://github.com/huggingface/tokenizers/pull/1308
* Derive clone for TrainerWrapper by @jonatanklosko in https://github.com/huggingface/tokenizers/pull/1317
* CD backports by @chris-ha458 in https://github.com/huggingface/tokenizers/pull/1318
* 0.13.4.rc1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1319

## New Contributors
* @sbhavani made their first contribution in https://github.com/huggingface/tokenizers/pull/1272
* @boyleconnor made their first contribution in https://github.com/huggingface/tokenizers/pull/1275
* @hiroshi-matsuda-rit made their first contribution in https://github.com/huggingface/tokenizers/pull/1289
* @bact made their first contribution in https://github.com/huggingface/tokenizers/pull/1292
* @kellymarchisio made their first contribution in https://github.com/huggingface/tokenizers/pull/1295
* @SamuelLarkin made their first contribution in https://github.com/huggingface/tokenizers/pull/1302
* @jonatanklosko made their first contribution in https://github.com/huggingface/tokenizers/pull/1317

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.4-rc2...v0.13.4.rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.4.rc1)

---

## v0.13.4-rc2: None
**Published:** 2023-05-17
**Pre-release**

None

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.4-rc2)

---

## v0.13.4-rc1: None
**Published:** 2023-05-15
**Pre-release**

None

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.4-rc1)

---

## v0.13.3: Rust v0.13.3
**Published:** 2023-04-05

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199
* New version 0.13.3 by @Narsil in https://github.com/huggingface/tokenizers/pull/1205
* New release  by @ArthurZucker in https://github.com/huggingface/tokenizers/pull/1207

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/v0.13.2...v0.13.3

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.3)

---

## python-v0.13.3: Python v0.13.3
**Published:** 2023-04-05

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/node-v0.13.2...python-v0.13.3rc1

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199
* New version 0.13.3 by @Narsil in https://github.com/huggingface/tokenizers/pull/1205

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/node-v0.13.2...python-v0.13.3rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.13.3)

---

## node-v0.13.3: Node v0.13.3
**Published:** 2023-04-05
**Pre-release**

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/node-v0.13.2...python-v0.13.3rc1

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199
* New version 0.13.3 by @Narsil in https://github.com/huggingface/tokenizers/pull/1205

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/node-v0.13.2...python-v0.13.3rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.13.3)

---

## python-v0.13.3rc1: Python v0.13.3rc1
**Published:** 2023-04-04
**Pre-release**

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/node-v0.13.2...python-v0.13.3rc1

## What's Changed
* Update pr docs actions by @mishig25 in https://github.com/huggingface/tokenizers/pull/1101
* Adding rust audit. by @Narsil in https://github.com/huggingface/tokenizers/pull/1099
* Revert "Update pr docs actions" by @mishig25 in https://github.com/huggingface/tokenizers/pull/1107
* Bump loader-utils from 1.4.0 to 1.4.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1108
* Include license file in Rust crate by @ankane in https://github.com/huggingface/tokenizers/pull/1115
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1116
* [FIX] In SentencePieceBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1120
* Fixing conda ssl location by @Narsil in https://github.com/huggingface/tokenizers/pull/1124
* Adding stale bot ? by @Narsil in https://github.com/huggingface/tokenizers/pull/1123
* Bump minimatch from 3.0.4 to 3.1.2 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1126
* Bump decode-uri-component from 0.2.0 to 0.2.2 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1125
* Bump cached-path from 0.5 to 0.6 by @hvaara in https://github.com/huggingface/tokenizers/pull/1127
* Wrap rustdoc html entity in code block by @hvaara in https://github.com/huggingface/tokenizers/pull/1130
* Fix broken links in docs by @hvaara in https://github.com/huggingface/tokenizers/pull/1133
* Bump derive_builder from 0.9 to 0.12 by @hvaara in https://github.com/huggingface/tokenizers/pull/1129
* Ignore Cargo.lock for subfolders by @hvaara in https://github.com/huggingface/tokenizers/pull/1131
* Fix one char super tiny typo by @fzyzcjy in https://github.com/huggingface/tokenizers/pull/1137
* [FIX] In CharBPETokenizer, when Vocab or merges is None, unk_token cannot be used. by @SeongBeomLEE in https://github.com/huggingface/tokenizers/pull/1136
* Bump json5, copy-webpack-plugin, webpack and webpack-cli in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1139
* Bump json5 from 2.2.0 to 2.2.3 in /bindings/node by @dependabot in https://github.com/huggingface/tokenizers/pull/1140
* Add missing build targets by @Narsil in https://github.com/huggingface/tokenizers/pull/1145
* Adding python 3.8 for M1 by @Narsil in https://github.com/huggingface/tokenizers/pull/1147
* Made dirs optional by @ankane in https://github.com/huggingface/tokenizers/pull/1148
* Update info on environment variable for threading by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1150
* Making `Tokenizer` clone. by @Narsil in https://github.com/huggingface/tokenizers/pull/1152
* Prevent using `from_pretrained` on invalid ids (better error message). by @Narsil in https://github.com/huggingface/tokenizers/pull/1153
* Improved version. by @Narsil in https://github.com/huggingface/tokenizers/pull/1154
* Update model.rs by @thomasw21 in https://github.com/huggingface/tokenizers/pull/1166
* Using clippy 1.67 by @Narsil in https://github.com/huggingface/tokenizers/pull/1167
* pyo3 v0.18 migration by @mert-kurttutan in https://github.com/huggingface/tokenizers/pull/1173
* Bump webpack from 5.75.0 to 5.76.0 in /tokenizers/examples/unstable_wasm/www by @dependabot in https://github.com/huggingface/tokenizers/pull/1181
* Fixing infinite loop in UnigramTrainer. by @Narsil in https://github.com/huggingface/tokenizers/pull/1182
* Bump dirs from 3.0 to 4.0 by @hvaara in https://github.com/huggingface/tokenizers/pull/1142
* Adding ByteFallback support for `tokenizers`. by @Narsil in https://github.com/huggingface/tokenizers/pull/1183
* Faster `datasets` train example by @lhoestq in https://github.com/huggingface/tokenizers/pull/1192
* Adding `Replace` to decoder (to undo the Replace Normalizer for Metaspace split). by @Narsil in https://github.com/huggingface/tokenizers/pull/1195
* Creating `normalizers.Prepend` (To be used instead of `Metaspace`). by @Narsil in https://github.com/huggingface/tokenizers/pull/1194
* Adding 2 new decoders: by @Narsil in https://github.com/huggingface/tokenizers/pull/1196
* Fixing decoder strip because of char boundaries. by @Narsil in https://github.com/huggingface/tokenizers/pull/1197
* Add `content` to Strip decoder to allow decoding mid tokens. by @Narsil in https://github.com/huggingface/tokenizers/pull/1199
* New version 0.13.3 by @Narsil in https://github.com/huggingface/tokenizers/pull/1205

## New Contributors
* @ankane made their first contribution in https://github.com/huggingface/tokenizers/pull/1115
* @SeongBeomLEE made their first contribution in https://github.com/huggingface/tokenizers/pull/1120
* @hvaara made their first contribution in https://github.com/huggingface/tokenizers/pull/1127
* @fzyzcjy made their first contribution in https://github.com/huggingface/tokenizers/pull/1137
* @mert-kurttutan made their first contribution in https://github.com/huggingface/tokenizers/pull/1150
* @lhoestq made their first contribution in https://github.com/huggingface/tokenizers/pull/1192

**Full Changelog**: https://github.com/huggingface/tokenizers/compare/node-v0.13.2...python-v0.13.3rc1

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.13.3rc1)

---

## v0.13.2: Rust 0.13.2
**Published:** 2022-11-07

Python 3.11 support (Python only modification)

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.2)

---

## python-v0.13.2: Python 0.13.2
**Published:** 2022-11-07

## [0.13.2] 

- [#1096] Python 3.11 support

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.13.2)

---

## node-v0.13.2: Node 0.13.2
**Published:** 2022-11-07
**Pre-release**

Python 3.11 support (Python only modification)

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.13.2)

---

## v0.13.1: Rust 0.13.1
**Published:** 2022-10-06

## [0.13.1]

- [#1072] Fixing Roberta type ids.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.1)

---

## python-v0.13.1: Python v0.13.1
**Published:** 2022-10-06

## [0.13.1]

- [#1072] Fixing Roberta type ids.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.13.1)

---

## node-v0.13.1: Node 0.13.1
**Published:** 2022-10-06

## [0.13.1]

- [#1072] Fixing Roberta type ids.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.13.1)

---

## python-v0.13.0: Python v0.13.0
**Published:** 2022-09-21

## [0.13.0] 

- [#956] PyO3 version upgrade
- [#1055] M1 automated builds
- [#1008] `Decoder` is now a composable trait, but without being backward incompatible
- [#1047, #1051, #1052] `Processor` is now a composable trait, but without being backward incompatible

Both trait changes warrant a "major" number since, despite best efforts to not break backward
  compatibility, the code is different enough that we cannot be exactly sure.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.13.0)

---

## v0.13.0: Rust v0.13.0
**Published:** 2022-09-19

## [0.13.0] 

- [#1009] `unstable_wasm` feature to support building on Wasm (it's unstable !)
- [#1008] `Decoder` is now a composable trait, but without being backward incompatible
- [#1047, #1051, #1052] `Processor` is now a composable trait, but without being backward incompatible

Both trait changes warrant a "major" number since, despite best efforts to not break backward
  compatibility, the code is different enough that we cannot be exactly sure.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.13.0)

---

## node-v0.13.0: Node v0.13.0
**Published:** 2022-09-19

## [0.13.0] 

- [#1008] `Decoder` is now a composable trait, but without being backward incompatible
- [#1047, #1051, #1052] `Processor` is now a composable trait, but without being backward incompatible

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.13.0)

---

## python-v0.12.1: Python v0.12.1
**Published:** 2022-04-13
**Pre-release**

## [0.12.1] 

- [#938] **Reverted breaking change**. https://github.com/huggingface/transformers/issues/16520

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.12.1)

---

## v0.12.0: [YANKED] Rust v0.12.0
**Published:** 2022-03-31

## [0.12.0]

Bump minor version because of a breaking change.

The breaking change was causing more issues upstream in `transformers` than anticipated:
https://github.com/huggingface/transformers/pull/16537#issuecomment-1085682657

The decision was to rollback on that breaking change, and figure out a different way later to do this modification

- [#938] **Breaking change**. Decoder trait is modified to be composable. This is only breaking if you are using decoders on their own. tokenizers should be error free.
- [#939] Making the regex in `ByteLevel` pre_tokenizer optional (necessary for BigScience)

- [#952] Fixed the vocabulary size of UnigramTrainer output (to respect added tokens)
- [#954] Fixed not being able to save vocabularies with holes in vocab (ConvBert). Yell warnings instead, but stop panicking.
- [#961] Added link for Ruby port of `tokenizers`
- [#960] Feature gate for `cli` and its `clap` dependency

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.12.0)

---

## python-v0.12.0:  [YANKED] Python v0.12.0
**Published:** 2022-03-31

## [0.12.0]

The breaking change was causing more issues upstream in `transformers` than anticipated:
https://github.com/huggingface/transformers/pull/16537#issuecomment-1085682657

The decision was to rollback on that breaking change, and figure out a different way later to do this modification

Bump minor version because of a breaking change.

- [#938] **Breaking change**. Decoder trait is modified to be composable. This is only breaking if you are using decoders on their own. tokenizers should be error free.
- [#939] Making the regex in `ByteLevel` pre_tokenizer optional (necessary for BigScience)

- [#952] Fixed the vocabulary size of UnigramTrainer output (to respect added tokens)
- [#954] Fixed not being able to save vocabularies with holes in vocab (ConvBert). Yell warnings instead, but stop panicking.
- [#962] Fix tests for python 3.10
- [#961] Added link for Ruby port of `tokenizers`

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.12.0)

---

## node-v0.12.0: [YANKED] Node v0.12.0 
**Published:** 2022-03-31

## [0.12.0]

The breaking change was causing more issues upstream in `transformers` than anticipated:
https://github.com/huggingface/transformers/pull/16537#issuecomment-1085682657

The decision was to rollback on that breaking change, and figure out a different way later to do this modification


Bump minor version because of a breaking change.
Using `0.12` to match other bindings.

- [#938] **Breaking change**. Decoder trait is modified to be composable. This is only breaking if you are using decoders on their own. tokenizers should be error free.
- [#939] Making the regex in `ByteLevel` pre_tokenizer optional (necessary for BigScience)

- [#952] Fixed the vocabulary size of UnigramTrainer output (to respect added tokens)
- [#954] Fixed not being able to save vocabularies with holes in vocab (ConvBert). Yell warnings instead, but stop panicking.
- [#961] Added link for Ruby port of `tokenizers`

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.12.0)

---

## v0.11.2: Rust v0.11.2
**Published:** 2022-02-28

- [#919] Fixing single_word AddedToken. (regression from 0.11.2)
- [#916] Deserializing faster `added_tokens` by loading them in batch.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.11.2)

---

## python-v0.11.6: Python v0.11.6
**Published:** 2022-02-28

- [#919] Fixing single_word AddedToken. (regression from 0.11.2)
- [#916] Deserializing faster `added_tokens` by loading them in batch.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.6)

---

## node-v0.8.3: Node v0.8.3
**Published:** 2022-02-28



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.8.3)

---

## python-v0.11.5: Python v0.11.5
**Published:** 2022-02-16

[#895] Add wheel support for Python 3.10

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.5)

---

## v0.11.1: Rust v0.11.1
**Published:** 2022-01-17

 - [#882] Fixing Punctuation deserialize without argument.
 - [#868] Fixing missing direction in TruncationParams
 - [#860] Adding TruncationSide to TruncationParams

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.11.1)

---

## python-v0.11.3: Python v0.11.3
**Published:** 2022-01-17

 - [#882] Fixing Punctuation deserialize without argument.
 - [#868] Fixing missing direction in TruncationParams
 - [#860] Adding TruncationSide to TruncationParams

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.3)

---

## node-v0.8.2: Node v0.8.2
**Published:** 2022-01-17

[#884] Fixing bad deserialization following inclusion of a default for Punctuation

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.8.2)

---

## node-v0.8.1: Node v0.8.1
**Published:** 2022-01-17

Fixing various backward compatibility bugs (Old serialized files couldn't be deserialized anymore.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.8.1)

---

## python-v0.11.4: Python v0.11.4
**Published:** 2022-01-17
**Pre-release**

[#884] Fixing bad deserialization following inclusion of a default for Punctuation

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.4)

---

## python-v0.11.2: Python v0.11.2
**Published:** 2022-01-04
**Pre-release**

Fixes https://github.com/huggingface/tokenizers/pull/868 

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.2)

---

## python-v0.11.1: Python v0.11.1
**Published:** 2021-12-28

[#860] Adding `TruncationSide` to `TruncationParams`.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.1)

---

## python-v0.11.0: Python v0.11.0
**Published:** 2021-12-24

### Fixed

- [#585] Conda version should now work on old CentOS
- [#844] Fixing interaction between `is_pretokenized` and `trim_offsets`.
- [#851] Doc links

### Added
- [#657]: Add SplitDelimiterBehavior customization to Punctuation constructor
- [#845]: Documentation for `Decoders`.

### Changed
- [#850]: Added a feature gate to enable disabling `http` features
- [#718]: Fix `WordLevel` tokenizer determinism during training
- [#762]: Add a way to specify the unknown token in `SentencePieceUnigramTokenizer`
- [#770]: Improved documentation for `UnigramTrainer`
- [#780]: Add `Tokenizer.from_pretrained` to load tokenizers from the Hugging Face Hub
- [#793]: Saving a pretty JSON file by default when saving a tokenizer

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.0)

---

## node-v0.8.0: Node v0.8.0
**Published:** 2021-09-02

### BREACKING CHANGES
- Many improvements on the Trainer ([#519](https://github.com/huggingface/tokenizers/pull/519)).
The files must now be provided first when calling `tokenizer.train(files, trainer)`.

### Features
- Adding the `TemplateProcessing`
- Add `WordLevel` and `Unigram` models ([#490](https://github.com/huggingface/tokenizers/pull/490))
- Add `nmtNormalizer` and `precompiledNormalizer` normalizers ([#490](https://github.com/huggingface/tokenizers/pull/490))
- Add `templateProcessing` post-processor ([#490](https://github.com/huggingface/tokenizers/pull/490))
- Add `digitsPreTokenizer` pre-tokenizer ([#490](https://github.com/huggingface/tokenizers/pull/490))
- Add support for mapping to sequences ([#506](https://github.com/huggingface/tokenizers/pull/506))
- Add `splitPreTokenizer` pre-tokenizer ([#542](https://github.com/huggingface/tokenizers/pull/542))
- Add `behavior` option to the `punctuationPreTokenizer` ([#657](https://github.com/huggingface/tokenizers/pull/657))
- Add the ability to load tokenizers from the Hugging Face Hub using `fromPretrained` ([#780](https://github.com/huggingface/tokenizers/pull/780))

### Fixes
- Fix a bug where long tokenizer.json files would be incorrectly deserialized ([#459](https://github.com/huggingface/tokenizers/pull/459))
- Fix RobertaProcessing deserialization in PostProcessorWrapper ([#464](https://github.com/huggingface/tokenizers/pull/464))

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/node-v0.8.0)

---

## python-v0.10.3: Python v0.10.3
**Published:** 2021-05-24

### Fixed
- [#686]: Fix SPM conversion process for whitespace deduplication
- [#707]: Fix stripping strings containing Unicode characters

### Added
- [#693]: Add a CTC Decoder for Wave2Vec models

### Removed
- [#714]: Removed support for Python 3.5

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.10.3)

---

## python-v0.10.2: Python v0.10.2
**Published:** 2021-04-05

### Fixed
- [#652]: Fix offsets for `Precompiled` corner case
- [#656]: Fix BPE `continuing_subword_prefix`
- [#674]: Fix `Metaspace` serialization problems

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.10.2)

---

## python-v0.10.1: Python v0.10.1
**Published:** 2021-02-04

### Fixed
- [#616]: Fix SentencePiece tokenizers conversion
- [#617]: Fix offsets produced by Precompiled Normalizer (used by tokenizers converted from SPM)
- [#618]: Fix Normalizer.normalize with `PyNormalizedStringRefMut`
- [#620]: Fix serialization/deserialization for overlapping models
- [#621]: Fix `ByteLevel` instantiation from a previously saved state (using `__getstate__()`)

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.10.1)

---

## python-v0.10.0: Python v0.10.0
**Published:** 2021-01-12

### Added
- [#508]: Add a Visualizer for notebooks to help understand how the tokenizers work
- [#519]: Add a `WordLevelTrainer` used to train a `WordLevel` model
- [#533]: Add support for conda builds
- [#542]: Add Split pre-tokenizer to easily split using a pattern
- [#544]: Ability to train from memory. This also improves the integration with `datasets`
- [#590]: Add getters/setters for components on BaseTokenizer
- [#574]: Add `fust_unk` option to SentencePieceBPETokenizer

### Changed
- [#509]: Automatically stubbing the `.pyi` files
- [#519]: Each `Model` can return its associated `Trainer` with `get_trainer()`
- [#530]: The various attributes on each component can be get/set (ie.
`tokenizer.model.dropout = 0.1`)
- [#538]: The API Reference has been improved and is now up-to-date.

## Fixed
- [#519]: During training, the `Model` is now trained in-place. This fixes several bugs that were
forcing to reload the `Model` after a training.
- [#539]: Fix `BaseTokenizer` enable_truncation docstring

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.10.0)

---

## python-v0.10.0rc1: Python v0.10.0rc1
**Published:** 2020-12-08
**Pre-release**

### Added
- [#508]: Add a Visualizer for notebooks to help understand how the tokenizers work
- [#519]: Add a `WordLevelTrainer` used to train a `WordLevel` model
- [#533]: Add support for conda builds
- [#542]: Add Split pre-tokenizer to easily split using a pattern
- [#544]: Ability to train from memory. This also improves the integration with `datasets`

### Changed
- [#509]: Automatically stubbing the `.pyi` files
- [#519]: Each `Model` can return its associated `Trainer` with `get_trainer()`
- [#530]: The various attributes on each component can be get/set (ie.
`tokenizer.model.dropout = 0.1`)
- [#538]: The API Reference has been improved and is now up-to-date.

## Fixed
- [#519]: During training, the `Model` is now trained in-place. This fixes several bugs that were
forcing to reload the `Model` after a training.
- [#539]: Fix `BaseTokenizer` enable_truncation docstring

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.10.0rc1)

---

## python-v0.9.4: Python v0.9.4
**Published:** 2020-11-10

### Fixed
- [#492]: Fix `from_file` on `BertWordPieceTokenizer`
- [#498]: Fix the link to download `sentencepiece_model_pb2.py`
- [#500]: Fix a typo in the docs quicktour

### Changed
- [#506]: Improve Encoding mappings for pairs of sequence

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.9.4)

---

## python-v0.9.3: Python v0.9.3
**Published:** 2020-10-26

### Fixed
- [#470]: Fix hanging error when training with custom component
- [#476]: TemplateProcessing serialization is now deterministic
- [#481]: Fix SentencePieceBPETokenizer.from_files

### Added
- [#477]: UnicodeScripts PreTokenizer to avoid merges between various scripts
- [#480]: Unigram now accepts an `initial_alphabet` and handles `special_tokens` correctly

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.9.3)

---

## python-v0.9.2: Python v0.9.2
**Published:** 2020-10-15

### Fixed
- [#464] Fix a problem with RobertaProcessing being deserialized as BertProcessing

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.9.2)

---

## python-v0.9.1: Python v0.9.1
**Published:** 2020-10-13

### Fixed
- [#459] Fix a problem with deserialization

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.9.1)

---

## python-v0.9.0: Python v0.9.0
**Published:** 2020-10-09

### Fixed
- [#362]: Fix training deadlock with Python components.
- [#363]: Fix a crash when calling `.train` with some non-existent files
- [#355]: Remove a lot of possible crashes
- [#389]: Improve truncation (crash and consistency)

### Added
- [#379]: Add the ability to call `encode`/`encode_batch` with numpy arrays
- [#292]: Support for the Unigram algorithm
- [#378], [#394], [#416], [#417]: Many new Normalizer and PreTokenizer
- [#403]: Add `TemplateProcessing` `PostProcessor`.
- [#420]: Ability to fuse the "unk" token in BPE.

### Changed
- [#360]: Lots of improvements related to words/alignment tracking
- [#426]: Improvements on error messages thanks to PyO3 0.12

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.9.0)

---

## python-v0.9.0.rc1: Python v0.9.0.rc1
**Published:** 2020-09-24
**Pre-release**

### Fixed
- [#362]: Fix training deadlock with Python components.
- [#363]: Fix a crash when calling `.train` with some non-existent files
- [#355]: Remove a lot of possible crashes
- [#389]: Improve truncation (crash and consistency)

### Added
- [#379]: Add the ability to call `encode`/`encode_batch` with numpy arrays
- [#292]: Support for the Unigram algorithm
- [#378], [#394], [#416], [#417]: Many new Normalizer and PreTokenizer
- [#403]: Add `TemplateProcessing` `PostProcessor`.
- [#420]: Ability to fuse the "unk" token in BPE.

### Changed
- [#360]: Lots of improvements related to words/alignment tracking
- [#426]: Improvements on error messages thanks to PyO3 0.12

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.9.0.rc1)

---

## python-v0.8.1: Python 0.8.1
**Published:** 2020-07-20

### Fixed
- [#333]: Fix deserialization of `AddedToken`, where the content was not restored properly

### Changed
- [#329]: Improved warning and behavior when we detect a fork
- [#330]: BertNormalizer now keeps the same behavior than the original implementation when
`strip_accents` is not specified.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.8.1)

---

## python-v0.8.0: Python v0.8.0
**Published:** 2020-06-26

### Highlights of this release
- We can now encode both pre-tokenized inputs, and raw strings. This is especially usefull when
processing datasets that are already pre-tokenized like for NER (Name Entity Recognition), and helps
while applying labels to each word.
- Full tokenizer serialization. It is now easy to save a tokenizer to a single JSON file, to later
load it back with just one line of code. That's what sharing a Tokenizer means now: 1 line of code.
- With the serialization comes the compatibility with `Pickle`! The Tokenizer, all of its components,
Encodings, everything can be pickled!
- Training a tokenizer is now even faster (up to 5-10x) than before!
- Compatibility with `multiprocessing`, even when using the `fork` start method. Since this library
makes heavy use of the multithreading capacities of our computers to allows a very fast tokenization,
this led to problems (deadlocks) when used with `multiprocessing`. This version now allows to
disable the parallelism, and will warn you if this is necessary.
- And a lot of other improvements, and fixes.

### Fixed
- [#286]: Fix various crash when training a BPE model
- [#309]: Fixed a few bugs related to additional vocabulary/tokens

### Added
- [#272]: Serialization of the `Tokenizer` and all the parts (`PreTokenizer`, `Normalizer`, ...).
This adds some methods to easily save/load an entire tokenizer (`from_str`, `from_file`).
- [#273]: `Tokenizer` and its parts are now pickable
- [#289]: Ability to pad to a multiple of a specified value. This is especially useful to ensure
activation of the Tensor Cores, while ensuring padding to a multiple of 8. Use with
`enable_padding(pad_to_multiple_of=8)` for example.
- [#298]: Ability to get the currently set truncation/padding params
- [#311]: Ability to enable/disable the parallelism using the `TOKENIZERS_PARALLELISM` environment
variable. This is especially usefull when using `multiprocessing` capabilities, with the `fork`
start method, which happens to be the default on Linux systems. Without disabling the parallelism,
the process dead-locks while encoding. (Cf [#187] for more information)

### Changed
- Improved errors generated during truncation: When the provided max length is too low are
now handled properly.
- [#249] `encode` and `encode_batch` now accept pre-tokenized inputs. When the input is pre-tokenized,
the argument `is_pretokenized=True` must be specified.
- [#276]: Improve BPE training speeds, by reading files sequentially, but parallelizing the
processing of each file
- [#280]: Use `onig` for byte-level pre-tokenization to remove all the differences with the original
implementation from GPT-2
- [#309]: Improved the management of the additional vocabulary. This introduces an option
`normalized`, controlling whether a token should be extracted from the normalized version of the
input text.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.8.0)

---

## python-v0.7.0: Python v0.7.0
**Published:** 2020-04-08

### Changed
- Only one progress bar while reading files during training. This is better for use-cases with
a high number of files as it avoids having too many progress bars on screen. Also avoids reading the
size of each file before starting to actually read these files, as this process could take really
long.
- [#193]: `encode` and `encode_batch` now take a new optional argument, specifying whether we
should add the special tokens. This is activated by default.
- [#197]: `original_str` and `normalized_str` have been removed from the `Encoding` returned by
`encode` and `encode_batch`. This brings a reduction of 70% of the memory footprint.
- [#197]: The offsets provided on `Encoding` are now relative to the original string, and not the
normalized one anymore.
- The added token given to `add_special_tokens` or `add_tokens` on a `Tokenizer`, or while using
`train(special_tokens=...)` can now be instances of `AddedToken` to provide more control over these
tokens.
- [#136]: Updated Pyo3 version
- [#136]: Static methods `Model.from_files` and `Model.empty` are removed in favor of using
constructors.
- [#239]: `CharBPETokenizer` now corresponds to OpenAI GPT BPE implementation by default.

### Added
- [#188]: `ByteLevel` is also a `PostProcessor` now and handles trimming the offsets if activated.
This avoids the unintuitive inclusion of the whitespaces in the produced offsets, even if these
whitespaces are part of the actual token.
It has been added to `ByteLevelBPETokenizer` but it is off by default (`trim_offsets=False`).
- [#236]: `RobertaProcessing` also handles trimming the offsets.
- [#234]: New alignment mappings on the `Encoding`. Provide methods to easily convert between `char`
or `word` (input space) and `token` (output space).
- `post_process` can be called on the `Tokenizer`
- [#208]: Ability to retrieve the vocabulary from the `Tokenizer` with
`get_vocab(with_added_tokens: bool)`
- [#136] Models can now be instantiated through object constructors.

### Fixed
- [#193]: Fix some issues with the offsets being wrong with the `ByteLevel` BPE:
	- when `add_prefix_space=True`
	- [#156]: when a Unicode character gets split-up in multiple byte-level characters
- Fix a bug where offsets were wrong when there was any added tokens in the sequence being encoded.
- [#175]: Fix a bug that prevented the addition of more than a certain amount of tokens (even if
not advised, but that's not the question).
- [#205]: Trim the decoded string in `BPEDecoder` used by `CharBPETokenizer`

### How to migrate
- Add the `ByteLevel` `PostProcessor` to your byte-level BPE tokenizers if relevant. If you are
using `ByteLevelBPETokenizer`, this option is disabled by default (`trim_offsets=False`).
- `BertWordPieceTokenizer` option to `add_special_tokens` must now be given to `encode` or
`encode_batch`
- Access to the `original_str` on the `Encoding` has been removed. The original string is the input
of `encode` so it didn't make sense to keep it here.
- No need to call `original_str.offsets(offsets[N])` to convert offsets to the original string. They
are now relative to the original string by default.
- Access to the `normalized_str` on the `Encoding` has been removed. Can be retrieved by calling
`normalize(sequence)` on the `Tokenizer`
- Change `Model.from_files` and `Model.empty` to use constructor. The model constructor should take
the same arguments as the old methods. (ie `BPE(vocab, merges)` or `BPE()`)
- If you were using the `CharBPETokenizer` and want to keep the same behavior as before, set
`bert_normalizer=False` and `split_on_whitespace_only=True`.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.7.0)

---

## rust-v0.10.1: Rust v0.10.1
**Published:** 2020-04-09

### Fixed
- [#226]: Fix the word indexes when there are special tokens

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/rust-v0.10.1)

---

## rust-v0.10.0: Rust v0.10.0
**Published:** 2020-04-08

### Changed
- [#222]: All Tokenizer's subparts must now be `Send + Sync`

### Added
- [#208]: Ability to retrieve the vocabulary from the `Tokenizer` & `Model`

### Fixed
- [#205]: Trim the decoded string in `BPEDecoder`
- [b770f36]: Fix a bug with added tokens generated IDs

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/rust-v0.10.0)

---

## rust-v0.9.0: Rust v0.9.0
**Published:** 2020-03-26

### Changed
- Only one progress bar while reading files during training. This is better for use-cases with
a high number of files as it avoids having too many progress bars on screen. Also avoids reading the
size of each file before starting to actually read these files, as this process could take really
long.
- [#190]: Improved BPE and WordPiece builders
- [#193]: `encode` and `encode_batch` now take a new argument, specifying whether we should add the
special tokens
- [#197]: The `NormalizedString` has been removed from the `Encoding`. It is now possible to
retrieve it by calling `normalize` on the `Tokenizer`. This brings a reduction of 70% of the memory
footprint
- [#197]: The `NormalizedString` API has been improved. It is now possible to retrieve parts of both
strings using both "normalized" or "original" offsets
- [#197]: The offsets provided on `Encoding` are now relative to the original string, and not the
normalized one anymore
- `AddedToken` are now used for both `add_special_tokens` and `add_tokens`. Also, these AddedToken
have more options to allow various behaviors.

### Added
- [#188]: `impl PostProcessor for ByteLevel`: Handles trimming the offsets if activated. This avoids
the unintuitive inclusion of the whitespaces in the produced offsets, even if these whitespaces are
part of the actual token
- More alignment mappings on the `Encoding`.
- `post_process` can be called on the `Tokenizer`

### Fixed
- [#193]: Fix some issues with the offsets being wrong with the `ByteLevel` BPE:
	- when `add_prefix_space` is activated
	- [#156]: when a Unicode character gets split-up in multiple byte-level characters
- Fix a bug where offsets were wrong when there was any added tokens in the sequence being encoded.
- [#175]: Fix a bug that prevented the addition of more than a certain amount of tokens (even if not
advised, but that's not the question)

### How to migrate
- Add the `ByteLevel` `PostProcessor` to your byte-level BPE tokenizers if relevant.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/rust-v0.9.0)

---

## rust-v0.8.0: Rust v0.8.0
**Published:** 2020-03-02

## Changes:
- Big improvements in speed for BPE (Both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))

## Fixes:
- Do not open all files directly while training ([#163](https://github.com/huggingface/tokenizers/issues/163))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
- The `LongestFirst` truncation strategy had a bug ([#174](https://github.com/huggingface/tokenizers/issues/174))

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/rust-v0.8.0)

---

## python-v0.6.0: Python v0.6.0
**Published:** 2020-03-02

## Changes:
- Big improvements in speed for BPE (Both training and tokenization) ([#165](https://github.com/huggingface/tokenizers/pull/165))

## Fixes:
- Some default tokens were missing from `BertWordPieceTokenizer` (cf [#160](https://github.com/huggingface/tokenizers/issues/160))
- There was a bug in ByteLevel PreTokenizer that caused offsets to be wrong if a char got split up
in multiple bytes. (cf [#156](https://github.com/huggingface/tokenizers/pull/156))
- The `longest_first` truncation strategy had a bug ([#174](https://github.com/huggingface/tokenizers/issues/174))

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.6.0)

---

## python-v0.5.2: Python v0.5.2
**Published:** 2020-02-24

## Fixes:
- We introduced a bug related to the saving of the WordPiece model in 0.5.2: The `vocab.txt` file was named
`vocab.json`. This is now fixed.
- The `WordLevel` model was also saving its vocabulary in the wrong format.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.5.2)

---

## python-v0.5.1: Python v0.5.1
**Published:** 2020-02-24

## Changes:
- `name` argument is now optional when saving a `Model`'s vocabulary. When the name is not specified,
the files get a more generic naming, like `vocab.json` or `merges.txt`.

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.5.1)

---

## python-v0.5.0: Python v0.5.0
**Published:** 2020-02-18

## Changes:
- `BertWordPieceTokenizer` now cleans up some tokenization artifacts while decoding (cf #145)
- `ByteLevelBPETokenizer` now has `dropout` (thanks @colinclement with #149)
- Added a new `Strip` normalizer
- `do_lowercase` has been changed to `lowercase` for consistency between the different tokenizers. (Especially `ByteLevelBPETokenizer` and `CharBPETokenizer`)
- Expose `__len__` on `Encoding` (cf #139)
- Improved padding performances.

## Fixes:
- #145: Decoding was buggy on `BertWordPieceTokenizer`.
- #152: Some documentation and examples were still using the old `BPETokenizer`


[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.5.0)

---

## python-v0.4.2: Python v0.4.2
**Published:** 2020-02-11

## Fixes:

- Fix a bug in the class `WordPieceTrainer` that prevented `BertWordPieceTokenizer` from being trained. (cf #137)

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.4.2)

---

## python-v0.4.1: Python v0.4.1
**Published:** 2020-02-11

## Fixes:

- Fix a bug related to the punctuation in BertWordPieceTokenizer (Thanks to @Mansterteddy with #134)

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.4.1)

---

## python-v0.4.0: Python v0.4.0
**Published:** 2020-02-10

## Changes:

- Replaced all `.new()` class methods by a proper `__new__` implementation. (Huge thanks to @ljos with #131)
- Improved typings

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.4.0)

---

## python-v0.3.0: Python v0.3.0
**Published:** 2020-02-05

## Changes:
- BPETokenizer has been renamed to CharBPETokenizer for clarity.
- Added `CharDelimiterSplit`: a new `PreTokenizer` that allows splitting sequences on the given delimiter (Works like `.split(delimiter)`)
- Added `WordLevel`: a new model that simply maps `tokens` to their `ids`.
- Improve truncation/padding and the handling of overflowing tokens. Now when a sequence gets truncated, we provide a list of overflowing `Encoding` that are ready to be processed by a language model, just as the main `Encoding`.
- Provide mapping to the original string offsets using:
```
output = tokenizer.encode(...)
print(output.original_str.offsets(output.offsets[3]))
```
- Exposed the vocabulary size on all tokenizers: https://github.com/huggingface/tokenizers/pull/99 by @kdexd

## Bug fixes:
- Fix a bug with IndexableString
- Fix a bug with truncation

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.3.0)

---

## python-v0.2.1: Python v0.2.1
**Published:** 2020-01-22

- Fix a bug with the IDs associated with added tokens.
- Fix a bug that was causing crashes in Python 3.5

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.2.1)

---

## python-v0.2.0: Python v0.2.0
**Published:** 2020-01-20

In this release, we fixed some inconsistencies between the `BPETokenizer` and the original python version of this tokenizer. If you created your own vocabulary using this Tokenizer, you will need to either train a new one, or use a modified version, where you set the `PreTokenizer` back to `Whitespace` (instead of `WhitespaceSplit`).

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/python-v0.2.0)

---

## v0.1.1: Python v0.1.1
**Published:** 2020-01-12

- Fix a bug where special tokens get split while encoding

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.1.1)

---

## v0.1.0: Python v0.1.0
**Published:** 2020-01-10



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.1.0)

---

## v0.0.13: v0.0.13
**Published:** 2020-01-08



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.13)

---

## v0.0.12: v0.0.12
**Published:** 2020-01-07



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.12)

---

## v0.0.11: v0.0.11
**Published:** 2019-12-27

Fixes the `sdist` build for Python

[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.11)

---

## v0.0.10: v0.0.10
**Published:** 2019-12-26



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.10)

---

## v0.0.9: v0.0.9
**Published:** 2019-12-23



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.9)

---

## v0.0.8: v0.0.8
**Published:** 2019-12-20



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.8)

---

## v0.0.7: v0.0.7
**Published:** 2019-12-17



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.7)

---

## v0.0.6: v0.0.6
**Published:** 2019-12-17



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.6)

---

## v0.0.5: v0.0.5
**Published:** 2019-12-13



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.5)

---

## v0.0.4: v0.0.4
**Published:** 2019-12-10



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.4)

---

## v0.0.3: v0.0.3
**Published:** 2019-12-03



[View on GitHub](https://github.com/huggingface/tokenizers/releases/tag/v0.0.3)

---

