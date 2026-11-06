# 发布与归档（投稿前最后一步）

## 1. 现状

| 项目 | 状态 |
|---|---|
| 仓库 | https://github.com/wyw-yy/medmem-law （**public**） |
| 分支 / 标签 | `master` / `v0.1.0`（附注标签，指向发布时的提交） |
| CI | `.github/workflows/ci.yml`：每次 push 跑「清单校验 + 审计 + 22 条声明核对 + 律重拟合 + base R 复算」 |
| 归档元数据 | `.zenodo.json`（已就位）、`CITATION.cff` |
| 数据/权重 | 不进仓库，用 `MANIFEST.data.sha256` + `docs/ENV.md` 留证 |

## 2. 拿 DOI（需要你的 Zenodo 账号，二选一）

**A. 网页集成（推荐，一次设置永久生效）**

1. 登录 <https://zenodo.org> → Settings → GitHub；
2. 在列表里找到 `wyw-yy/medmem-law`，把开关打开；
3. 回到 GitHub，`Releases → Draft a new release`，tag 填 `v0.1.0`，发布；
4. 几分钟后 Zenodo 会为该 release 生成 DOI，并在仓库里自动加 badge。

**B. 用 API token（我可以用命令行代做，需要你给 token）**

```powershell
# 你自己在 https://zenodo.org/account/settings/applications/ 建一个 token（scope: deposit:write）
$env:ZENODO_TOKEN = "..."
```

拿到 token 后我可以用它创建 deposition、上传 `medmem-law-v0.1.0.zip`（≈7 MB，不含数据）、提交并发 `.zenodo.json` 元数据，返回 DOI。

**C. 完全离线退路**：有些期刊接受 "code available at <commit hash>，永久镜像随投稿上传"，此时不需要 DOI；但审稿人会偏好 DOI。

## 3. 拿到 DOI 后要改的三处（我会一起做）

1. `CITATION.cff`：加 `doi: 10.5281/zenodo.XXXXXX`；
2. `README.md`：在标题下加 DOI badge（`[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)`）；
3. `docs/SUPPLEMENT.md` §5：写明"论文数字绑定 `v0.1.0` / DOI"。

## 4. 投稿前检查单

- [ ] 正文每个数字都能在 `docs/SUPPLEMENT.md` 的映射表里找到出处；
- [ ] `python -m tools.ci_check` 本地与 CI 都绿；
- [ ] `figs/*.pdf` 是矢量（Fig.1 三联图已同时出 png/pdf）；
- [ ] 三条未达标项（R²≥0.9 ×2、安全表 0/28）在正文 limitation 里出现，且措辞与 `docs/SUPPLEMENT.md` §2 一致；
- [ ] 数据可用性声明与许可一致（代码 MIT / 文档 CC BY 4.0 / 数据不转售）；
- [ ] 若期刊要求"代码匿名评审"：先建一个临时私有镜像仓库，评审结束后再切回公开 DOI。

## 5. 可直接用的英文段落

**Data availability**

> All experiments use public data only. We use MedMNIST v2 (CC BY 4.0) subsets
> — PathMNIST, BloodMNIST and PneumoniaMNIST — obtained as 224 px JPEGs from the
> public mirror `dz-osamu/MedMNIST` on Hugging Face and resized locally to 64 px;
> the 28 px variants were used for the pilot study. No institutional data were
> used in the results reported here. SHA-256 hashes of the exact dataset files
> used are provided in `MANIFEST.data.sha256`, and the preprocessing is a single
> reproducible command (`build/get_hf64.py`).

**Code availability**

> Code, the complete measurement table (one row per checkpoint), the fitted
> parameters with bootstrap intervals, and the audit scripts are available at
> https://github.com/wyw-yy/medmem-law (tag `v0.1.0`, DOI 10.5281/zenodo.XXXXXX).
> Every number in this manuscript is mapped to a file and a command in
> `docs/SUPPLEMENT.md`; `tools/ci_check.py` re-verifies the mapping on every
> commit, and an independent implementation of the fit in base R reproduces the
> residual sum of squares to within 3e-9.

**Ethics statement**

> This study did not involve human subjects or institutional patient data.
> All imaging data are public benchmark datasets. Because the manuscript's
> subject is memorization, we audited the public datasets themselves for exact
> and near-duplicate images across splits (`results/leak_report.json`);
> BloodMNIST was found to contain 5 exact and 4 near-duplicate pairs, which are
> excluded from threshold calibration.