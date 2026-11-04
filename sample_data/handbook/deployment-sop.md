# 正式環境部署 SOP

## 部署時段
- 一般部署：週二、週四 14:00–17:00。
- 週五、國定假日前一天 **禁止部署**，緊急修補需經技術長核准。

## 部署前檢查清單
1. Merge Request 至少一位 reviewer 核准，CI（單元測試、Lint、安全掃描）全部通過。
2. 在 staging 環境完成驗證，QA 於 Jira 單上標記「Staging Verified」。
3. 若有資料庫 migration，需先在 staging 演練，並準備 rollback script。
4. 在 #deploy 頻道公告預計部署時間與影響範圍。

## 部署步驟
1. 在 GitLab 建立 release tag（格式 vYYYY.MM.DD-N）。
2. 由 CI/CD pipeline 自動建置 Docker image 並推送到 Harbor。
3. 使用 Argo CD 同步到正式環境，採 rolling update。
4. 觀察 Grafana 儀表板 15 分鐘：錯誤率、p95 延遲、CPU/記憶體。

## Rollback
- 錯誤率超過 1% 或 p95 延遲增加 50% 以上，立即在 Argo CD 回滾到上一版。
- 回滾後在 #incident 頻道通報，並於 2 個工作天內完成事後檢討（postmortem）。
