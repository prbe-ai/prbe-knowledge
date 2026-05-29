{{/*
Expand the name of the chart.
*/}}
{{- define "probe-knowledge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name. Truncated at 63 chars for DNS-name limits.
*/}}
{{- define "probe-knowledge.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version label.
*/}}
{{- define "probe-knowledge.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "probe-knowledge.labels" -}}
helm.sh/chart: {{ include "probe-knowledge.chart" . }}
{{ include "probe-knowledge.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels (stable; never add version here).
*/}}
{{- define "probe-knowledge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "probe-knowledge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Per-component selector labels. Pass a dict: (dict "ctx" . "component" "ingestion").
*/}}
{{- define "probe-knowledge.componentSelectorLabels" -}}
{{ include "probe-knowledge.selectorLabels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Per-component labels (common + component).
*/}}
{{- define "probe-knowledge.componentLabels" -}}
{{ include "probe-knowledge.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
The image reference (repository:tag), defaulting tag to appVersion.
*/}}
{{- define "probe-knowledge.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end }}

{{/*
ServiceAccount name to use.
*/}}
{{- define "probe-knowledge.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "probe-knowledge.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Name of the Secret holding provider keys / tokens.
*/}}
{{- define "probe-knowledge.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "probe-knowledge.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Name of the non-secret ConfigMap.
*/}}
{{- define "probe-knowledge.configMapName" -}}
{{- printf "%s-config" (include "probe-knowledge.fullname" .) }}
{{- end }}

{{/*
Name of the bundled Postgres StatefulSet / Service / Secret.
*/}}
{{- define "probe-knowledge.postgresName" -}}
{{- printf "%s-postgres" (include "probe-knowledge.fullname" .) }}
{{- end }}

{{/*
The asyncpg DATABASE_URL. When the bundled Postgres is enabled, derive it from
the bundled auth; otherwise use postgresql.external.url.
*/}}
{{- define "probe-knowledge.databaseUrl" -}}
{{- if .Values.postgresql.bundled.enabled -}}
{{- $a := .Values.postgresql.bundled.auth -}}
{{- printf "postgresql://%s:%s@%s:5432/%s" $a.username $a.password (include "probe-knowledge.postgresName" .) $a.database -}}
{{- else -}}
{{- required "postgresql.external.url is required when postgresql.bundled.enabled=false" .Values.postgresql.external.url -}}
{{- end -}}
{{- end }}

{{/*
The alembic DATABASE_URL_SYNC (psycopg driver). Same source rules as above.
*/}}
{{- define "probe-knowledge.databaseUrlSync" -}}
{{- if .Values.postgresql.bundled.enabled -}}
{{- $a := .Values.postgresql.bundled.auth -}}
{{- printf "postgresql+psycopg://%s:%s@%s:5432/%s" $a.username $a.password (include "probe-knowledge.postgresName" .) $a.database -}}
{{- else -}}
{{- required "postgresql.external.urlSync is required when postgresql.bundled.enabled=false" .Values.postgresql.external.urlSync -}}
{{- end -}}
{{- end }}

{{/*
Common environment for every app role: non-secret config from the ConfigMap,
secret keys via secretKeyRef. Optional connector/gateway keys are wired only
when present in the rendered Secret.
*/}}
{{- define "probe-knowledge.env" -}}
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: DATABASE_URL
- name: DATABASE_URL_SYNC
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: DATABASE_URL_SYNC
- name: DEFAULT_CUSTOMER_ID
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: DEFAULT_CUSTOMER_ID
- name: ENVIRONMENT
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: ENVIRONMENT
- name: R2_ENDPOINT_URL
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: R2_ENDPOINT_URL
- name: R2_REGION
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: R2_REGION
- name: R2_BUCKET_PREFIX
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: R2_BUCKET_PREFIX
- name: R2_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: R2_ACCESS_KEY_ID
- name: R2_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: R2_SECRET_ACCESS_KEY
- name: GOOGLE_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: GOOGLE_API_KEY
- name: TOKEN_ENCRYPTION_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: TOKEN_ENCRYPTION_KEY
- name: KNOWLEDGE_API_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: KNOWLEDGE_API_TOKEN
{{- if .Values.secrets.create }}
{{- if .Values.secrets.anthropicApiKey }}
- name: ANTHROPIC_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: ANTHROPIC_API_KEY
{{- end }}
{{- if .Values.secrets.openaiApiKey }}
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: OPENAI_API_KEY
{{- end }}
{{- if .Values.secrets.llmGatewayKey }}
- name: LLM_GATEWAY_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" . }}
      key: LLM_GATEWAY_KEY
{{- end }}
{{- range $k, $v := .Values.secrets.connectors }}
{{- if $v }}
- name: {{ $k | snakecase | upper }}
  valueFrom:
    secretKeyRef:
      name: {{ include "probe-knowledge.secretName" $ }}
      key: {{ $k | snakecase | upper }}
{{- end }}
{{- end }}
{{- end }}
{{- if .Values.config.llmGatewayUrl }}
- name: LLM_GATEWAY_URL
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: LLM_GATEWAY_URL
{{- end }}
{{- if .Values.config.otlpEndpoint }}
- name: GRAFANA_OTLP_ENDPOINT
  valueFrom:
    configMapKeyRef:
      name: {{ include "probe-knowledge.configMapName" . }}
      key: GRAFANA_OTLP_ENDPOINT
{{- end }}
{{- range $k, $v := .Values.config.extraEnv }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- end }}
