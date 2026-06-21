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

variable "streamlit_allowed_cidr_blocks" {
  type        = set(string)
  default     = []
  description = "CIDR blocks allowed to reach the public Streamlit frontend on TCP 8501. Use [\"0.0.0.0/0\"] only for temporary MVP demos."

  validation {
    condition     = alltrue([for cidr in var.streamlit_allowed_cidr_blocks : can(cidrhost(cidr, 0))])
    error_message = "Every streamlit_allowed_cidr_blocks value must be a valid IPv4 CIDR block."
  }
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
