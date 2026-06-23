variable "aws_region" {
  type    = string
  default = "ap-northeast-3"
}

variable "project_name" {
  type    = string
  default = "llmenhance"
}

variable "environment" {
  type    = string
  default = "mvp-osaka"
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

variable "model_server_security_group_id" {
  type        = string
  default     = ""
  description = "Existing Ollama/Qwen model server security group. When set, Terraform allows app EC2 ingress to Ollama."
}

variable "model_server_ollama_port" {
  type        = number
  default     = 11434
  description = "Ollama TCP port on the existing model server."

  validation {
    condition     = var.model_server_ollama_port > 0 && var.model_server_ollama_port <= 65535
    error_message = "model_server_ollama_port must be a valid TCP port."
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

variable "llm_model" {
  type        = string
  default     = "qwen2.5:7b"
  description = "Ollama chat model name available on the existing model server."

  validation {
    condition     = length(trimspace(var.llm_model)) > 0
    error_message = "llm_model must be a non-empty Ollama model name."
  }
}

variable "embedding_model" {
  type        = string
  default     = "bge-m3"
  description = "Ollama embedding model name available on the existing model server."

  validation {
    condition     = length(trimspace(var.embedding_model)) > 0
    error_message = "embedding_model must be a non-empty Ollama model name."
  }
}

variable "ollama_base_url" {
  type = string

  validation {
    condition     = length(trimspace(var.ollama_base_url)) > 0 && (startswith(var.ollama_base_url, "http://") || startswith(var.ollama_base_url, "https://"))
    error_message = "ollama_base_url must be a non-empty http:// or https:// URL for the existing model server."
  }
}
