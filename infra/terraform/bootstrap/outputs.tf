output "state_bucket" {
  value = aws_s3_bucket.terraform_state.bucket
}

output "state_region" {
  value = var.aws_region
}
