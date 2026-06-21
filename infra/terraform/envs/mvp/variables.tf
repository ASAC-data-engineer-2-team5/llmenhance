variable "aws_region" {
  type    = string
  default = "ap-northeast-2"
}

variable "project_name" {
  type    = string
  default = "llmenhance"
}

variable "environment" {
  type    = string
  default = "mvp"
}

variable "vpc_id" {
  type    = string
  default = ""
}

variable "subnet_id" {
  type    = string
  default = ""
}

variable "ami_id" {
  type    = string
  default = ""
}

variable "instance_type" {
  type    = string
  default = "t3.large"
}

variable "root_volume_gb" {
  type    = number
  default = 100
}

variable "repo_url" {
  type    = string
  default = "https://github.com/ASAC-data-engineer-2-team5/llmenhance.git"
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "ollama_base_url" {
  type = string

  validation {
    condition     = length(trimspace(var.ollama_base_url)) > 0 && (startswith(var.ollama_base_url, "http://") || startswith(var.ollama_base_url, "https://"))
    error_message = "ollama_base_url must be a non-empty http:// or https:// URL for the existing model server."
  }
}
