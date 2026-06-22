terraform {
  backend "s3" {
    bucket       = "llmenhance-mvp-tfstate-placeholder"
    key          = "llmenhance/mvp/terraform.tfstate"
    region       = "ap-northeast-3"
    use_lockfile = true
  }
}
