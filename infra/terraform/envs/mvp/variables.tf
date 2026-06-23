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

variable "enable_gemini_endpoint" {
  type        = bool
  default     = false
  description = "Enable the optional Vertex Gemini comparison API endpoint."
}

variable "enable_gemini_panel" {
  type        = bool
  default     = false
  description = "Show the optional Vertex Gemini comparison panel in Streamlit."
}

variable "google_application_credentials" {
  type        = string
  default     = ""
  description = "Container path for Vertex Gemini credentials when Gemini is enabled."
}

variable "google_cloud_project" {
  type        = string
  default     = ""
  description = "Google Cloud project for optional Vertex Gemini comparison."
}

variable "google_cloud_location" {
  type        = string
  default     = "us-central1"
  description = "Google Cloud location for optional Vertex Gemini comparison."
}

variable "gemini_model" {
  type        = string
  default     = "gemini-2.5-flash"
  description = "Vertex Gemini model for optional comparison."
}

variable "gemini_thinking_budget" {
  type        = number
  default     = 0
  description = "Vertex Gemini thinking budget for optional comparison."
}

variable "enable_bedrock_endpoint" {
  type        = bool
  default     = false
  description = "Enable the optional AWS Bedrock comparison API endpoint."
}

variable "enable_bedrock_panel" {
  type        = bool
  default     = false
  description = "Show the optional AWS Bedrock comparison panel in Streamlit."
}

variable "bedrock_region" {
  type        = string
  default     = "ap-northeast-3"
  description = "AWS Bedrock region for optional comparison."
}

variable "bedrock_model_id" {
  type        = string
  default     = ""
  description = "AWS Bedrock model or inference profile id for optional comparison."
}

variable "bedrock_model_label" {
  type        = string
  default     = ""
  description = "Human-readable AWS Bedrock model label for optional comparison."
}

variable "bedrock_max_output_tokens" {
  type        = number
  default     = 512
  description = "Maximum output tokens for optional AWS Bedrock comparison."
}

variable "ollama_base_url" {
  type = string

  validation {
    condition     = length(trimspace(var.ollama_base_url)) > 0 && (startswith(var.ollama_base_url, "http://") || startswith(var.ollama_base_url, "https://"))
    error_message = "ollama_base_url must be a non-empty http:// or https:// URL for the existing model server."
  }
}
