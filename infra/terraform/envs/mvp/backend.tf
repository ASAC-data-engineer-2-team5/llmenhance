terraform {
  backend "s3" {
    key          = "llmenhance/mvp/terraform.tfstate"
    region       = "ap-northeast-2"
    use_lockfile = true
  }
}
