"use client"
import React, { useEffect, useRef, useState } from 'react'
import { Button, Input, Card, Textarea, Slider, Switch, CardHeader, CardBody, Divider, Select, SelectItem } from "@nextui-org/react";
import OpenAI from "openai"

import { EyeSlashFilledIcon } from './EyeSlashFilledIcon';
import { EyeFilledIcon } from './EyeFilledIcon';

import { FileUploadButton } from '@/components/FileUploadButton';
import { SubtitleCard } from '@/components/SubtitleCard';
import { downloadString } from '@/utils/download';
import { sampleSrt } from '@/data/sample';

import {
  Translator,
  TranslatorStructuredArray,
  subtitleParser,
  CooldownContext,
  OpenAIProvider,
  OllamaProvider,
  OllamaQwen3_32B,
  OllamaTranslateGemma12B,
  OllamaTranslateGemma4B,
} from "llm-subtitle-translator"

const STORAGE_PROVIDER = "PROVIDER"
const STORAGE_OPENAI_API_KEY = "OPENAI_API_KEY"
const STORAGE_OPENAI_BASE_URL = "OPENAI_BASE_URL"
const STORAGE_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
const STORAGE_RATE_LIMIT = "RATE_LIMIT"
const STORAGE_MODEL = "MODEL"

const PROVIDERS = [
  { key: "openai", label: "OpenAI" },
  { key: "ollama", label: "Ollama" },
]

const DEFAULT_MODELS = {
  openai: "gpt-4o-mini",
  ollama: "",
}

const DefaultTemperature = 0

/** Map well-known Ollama model names to their specific provider classes */
function createOllamaProviderForModel(modelName, baseURL) {
  if (modelName === 'qwen3:32b') return new OllamaQwen3_32B(baseURL, true)
  if (modelName === 'translategemma:12b') return new OllamaTranslateGemma12B(baseURL, true)
  if (modelName === 'translategemma:4b') return new OllamaTranslateGemma4B(baseURL, true)
  return new OllamaProvider({ model: modelName, baseURL, dangerouslyAllowBrowser: true })
}

function RefreshIcon({ className }) {
  return (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
    </svg>
  )
}

export function TranslatorApplication() {
  // Translator Configuration
  const [providerKey, setProviderKey] = useState("openai")
  const [APIvalue, setAPIValue] = useState("")
  const [openaiBaseUrl, setOpenaiBaseUrlValue] = useState(undefined)
  const [ollamaBaseUrl, setOllamaBaseUrlValue] = useState(undefined)
  const [fromLanguage, setFromLanguage] = useState("")
  const [toLanguage, setToLanguage] = useState("English")
  const [systemInstruction, setSystemInstruction] = useState("")
  const [model, setModel] = useState(DEFAULT_MODELS["openai"])
  const [temperature, setTemperature] = useState(DefaultTemperature)
  const [batchSizes, setBatchSizes] = useState([10, 50])
  const [useStructuredMode, setUseStructuredMode] = useState(true)
  const [rateLimit, setRateLimit] = useState(60)

  const [isAPIInputVisible, setIsAPIInputVisible] = useState(false)
  const toggleAPIInputVisibility = () => setIsAPIInputVisible(!isAPIInputVisible)
  const [mounted, setMounted] = useState(false)

  // Model list
  const [availableModels, setAvailableModels] = useState([])
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState("")

  // Translator State
  const [srtInputText, setSrtInputText] = useState(sampleSrt)
  const [srtOutputText, setSrtOutputText] = useState(sampleSrt)
  const [inputs, setInputs] = useState(subtitleParser.fromSrt(sampleSrt).map(x => x.text))
  const [outputs, setOutput] = useState([])
  const [streamOutput, setStreamOutput] = useState("")
  const [translatorRunningState, setTranslatorRunningState] = useState(false)
  /** @type {React.RefObject<Translator>} */
  const translatorRef = useRef(null)
  const translatorRunningRef = useRef(false)

  // Translator Stats
  const [usageInformation, setUsageInformation] = useState(/** @type {typeof Translator.prototype.usage}*/(null))
  const [RPMInfomation, setRPMInformation] = useState(0)

  // Persistent Data Restoration
  useEffect(() => {
    const savedProvider = localStorage.getItem(STORAGE_PROVIDER) ?? "openai"
    const savedOllamaUrl = localStorage.getItem(STORAGE_OLLAMA_BASE_URL) ?? undefined
    setProviderKey(savedProvider)
    setAPIValue(localStorage.getItem(STORAGE_OPENAI_API_KEY) ?? "")
    setRateLimit(Number(localStorage.getItem(STORAGE_RATE_LIMIT) ?? rateLimit))
    setOpenaiBaseUrlValue(localStorage.getItem(STORAGE_OPENAI_BASE_URL) ?? undefined)
    setOllamaBaseUrlValue(savedOllamaUrl)
    const savedModel = localStorage.getItem(STORAGE_MODEL)
    setModel(savedModel ?? DEFAULT_MODELS[savedProvider] ?? "")
    setMounted(true)
    fetchModels(savedProvider, savedOllamaUrl)
  }, [])

  function handleProviderChange(key) {
    if (!key) return
    localStorage.setItem(STORAGE_PROVIDER, key)
    setProviderKey(key)
    setAvailableModels([])
    const defaultModel = DEFAULT_MODELS[key] ?? ""
    setModel(defaultModel)
    setModelValue(defaultModel)
    setUseStructuredMode(key === "openai")
    if (key === "ollama") {
      fetchModels(key)
    }
  }

  function setAPIKey(value) {
    localStorage.setItem(STORAGE_OPENAI_API_KEY, value)
    setAPIValue(value)
  }

  function setOpenaiBaseUrl(value) {
    if (!value) {
      value = undefined
      localStorage.removeItem(STORAGE_OPENAI_BASE_URL)
    } else {
      localStorage.setItem(STORAGE_OPENAI_BASE_URL, value)
    }
    setOpenaiBaseUrlValue(value)
  }

  function setOllamaBaseUrl(value) {
    if (!value) {
      value = undefined
      localStorage.removeItem(STORAGE_OLLAMA_BASE_URL)
    } else {
      localStorage.setItem(STORAGE_OLLAMA_BASE_URL, value)
    }
    setOllamaBaseUrlValue(value)
  }

  /**
   * @param {string} value
   */
  function setRateLimitValue(value) {
    localStorage.setItem(STORAGE_RATE_LIMIT, value)
    setRateLimit(Number(value))
  }

  /**
   * @param {string | undefined} value
   */
  function setModelValue(value) {
    if (!value) {
      localStorage.removeItem(STORAGE_MODEL)
    }
    else {
      localStorage.setItem(STORAGE_MODEL, value)
    }
    setModel(value ?? "")
  }

  async function fetchModels(overrideProvider, overrideOllamaUrl) {
    const provider = overrideProvider ?? providerKey
    const ollamaUrl = overrideOllamaUrl !== undefined ? overrideOllamaUrl : ollamaBaseUrl
    setModelsLoading(true)
    setModelsError("")
    try {
      let client
      if (provider === "openai") {
        client = new OpenAI({
          apiKey: APIvalue,
          baseURL: openaiBaseUrl,
          dangerouslyAllowBrowser: true,
        })
      } else {
        client = new OpenAI({
          apiKey: "ollama",
          baseURL: ollamaUrl ?? "http://localhost:11434/v1",
          dangerouslyAllowBrowser: true,
        })
      }
      const response = await client.models.list()
      const models = []
      for await (const m of response) {
        models.push(m.id)
      }
      models.sort()
      setAvailableModels(models)
    } catch (error) {
      console.error("[UI] Failed to fetch models:", error)
      const msg = error?.message ?? String(error)
      if (provider === "ollama" && (msg.includes("Connection error") || msg.includes("NetworkError") || msg.includes("Failed to fetch"))) {
        setModelsError(
          "CORS blocked. Start Ollama with: OLLAMA_ORIGINS=" + window.location.origin + " ollama serve"
        )
      } else {
        setModelsError("Failed to fetch models: " + msg)
      }
    } finally {
      setModelsLoading(false)
    }
  }

  function createProvider() {
    if (providerKey === "openai") {
      return new OpenAIProvider({
        apiKey: APIvalue,
        dangerouslyAllowBrowser: true,
        baseURL: openaiBaseUrl,
      }, model || undefined)
    } else {
      return createOllamaProviderForModel(model, ollamaBaseUrl)
    }
  }

  function canStart() {
    if (translatorRunningState) return false
    if (providerKey === "openai" && !APIvalue) return false
    if (providerKey === "ollama" && !model) return false
    return true
  }

  async function generate(e) {
    e.preventDefault()
    setTranslatorRunningState(true)
    console.log("[User Interface]", "Begin Generation")
    translatorRunningRef.current = true
    setOutput([])
    setUsageInformation(null)
    let currentStream = ""
    const outputWorkingProgress = subtitleParser.fromSrt(srtInputText)
    const currentOutputs = []

    const provider = createProvider()
    console.log("[User Interface]", "Provider:", provider.name, "Model:", model)

    const coolerChatGPTAPI = new CooldownContext(rateLimit, 60000, "ChatGPTAPI")

    const effectiveStructuredMode = useStructuredMode && provider.supportsStructuredOutput

    const TranslatorImplementation = effectiveStructuredMode ? TranslatorStructuredArray : Translator

    translatorRef.current = new TranslatorImplementation({ from: fromLanguage, to: toLanguage }, {
      provider,
      cooler: coolerChatGPTAPI,
      onStreamChunk: (data) => {
        if (currentStream === '' && data === "\n") {
          return
        }
        currentStream += data
        setStreamOutput(currentStream)
      },
      onStreamEnd: () => {
        currentStream = ""
        if (translatorRef.current?.aborted) {
          return
        }
        setStreamOutput("")
      },
      onClearLine: () => {
        const progressLines = currentStream.split("\n")
        if (progressLines[0] === "") {
          progressLines.shift()
        }
        progressLines.pop()
        currentStream = progressLines.join("\n") + "\n"
        if (currentStream === "\n") {
          currentStream = ""
        }
        setStreamOutput(currentStream)
      }
    }, {
      useModerator: false,
      batchSizes: batchSizes,
      createChatCompletionRequest: {
        model: model || provider.defaultModel,
        temperature: temperature,
        stream: true
      },
    })

    if (systemInstruction) {
      translatorRef.current.systemInstruction = systemInstruction
    }

    try {
      setStreamOutput("")
      for await (const output of translatorRef.current.translateLines(inputs)) {
        if (!translatorRunningRef.current) {
          console.error("[User Interface]", "Aborted")
          break
        }
        currentOutputs.push(output.finalTransform)
        const srtEntry = outputWorkingProgress[output.index - 1]
        srtEntry.text = output.finalTransform
        setOutput([...currentOutputs])
        setUsageInformation(translatorRef.current.usage)
        setRPMInformation(translatorRef.current.services.cooler?.rate)
      }
      console.log({ sourceInputWorkingCopy: outputWorkingProgress })
      setSrtOutputText(subtitleParser.toSrt(outputWorkingProgress))
    } catch (error) {
      console.error(error)
      alert(error?.message ?? error)
    }
    translatorRunningRef.current = false
    translatorRef.current = null
    setTranslatorRunningState(false)
  }

  async function stopGeneration() {
    console.error("[User Interface]", "Aborting")
    if (translatorRef.current) {
      translatorRunningRef.current = false
      translatorRef.current.abort()
    }
  }

  const isOllama = providerKey === "ollama"

  return (
    <>
      <div className='w-full'>
        <form id="translator-config-form" onSubmit={(e) => generate(e)}>
          <div className='px-4 pt-4 flex flex-wrap justify-between w-full gap-4'>
            <Card className="z-10 w-full shadow-md border" shadow="none">
              <CardHeader className="flex gap-3 pb-0">
                <div className="flex flex-col">
                  <p className="text-md">Configuration</p>
                </div>
              </CardHeader>
              <CardBody>
                <div className='flex flex-wrap justify-between w-full gap-4'>
                  <div className='flex flex-wrap md:flex-nowrap w-full gap-4'>
                    <Select
                      className="w-full md:w-4/12"
                      size='sm'
                      label="LLM Provider"
                      selectedKeys={mounted ? [providerKey] : []}
                      onSelectionChange={(keys) => handleProviderChange([...keys][0])}
                    >
                      {PROVIDERS.map((p) => (
                        <SelectItem key={p.key} value={p.key}>
                          {p.label}
                        </SelectItem>
                      ))}
                    </Select>

                    {mounted && !isOllama && (
                      <Input
                        className="w-full md:w-4/12"
                        size='sm'
                        value={APIvalue}
                        onValueChange={(value) => setAPIKey(value)}
                        isRequired
                        autoComplete='off'
                        label="API Key"
                        variant="flat"
                        description="Stored locally in browser"
                        endContent={
                          <button className="focus:outline-none" type="button" onClick={toggleAPIInputVisibility}>
                            {isAPIInputVisible ? (
                              <EyeSlashFilledIcon className="text-2xl text-default-400 pointer-events-none" />
                            ) : (
                              <EyeFilledIcon className="text-2xl text-default-400 pointer-events-none" />
                            )}
                          </button>
                        }
                        type={isAPIInputVisible ? "text" : "password"}
                      />
                    )}

                    {mounted && !isOllama && (
                      <Input
                        className='w-full md:w-4/12'
                        size='sm'
                        type="text"
                        label="Base URL"
                        placeholder="https://api.openai.com/v1"
                        autoComplete='on'
                        value={openaiBaseUrl ?? ""}
                        onValueChange={setOpenaiBaseUrl}
                      />
                    )}

                    {mounted && isOllama && (
                      <Input
                        className='w-full md:w-8/12'
                        size='sm'
                        type="text"
                        label="Ollama URL"
                        placeholder="http://localhost:11434/v1"
                        autoComplete='on'
                        value={ollamaBaseUrl ?? ""}
                        onValueChange={setOllamaBaseUrl}
                        description="Leave empty for default (localhost:11434)"
                      />
                    )}
                  </div>

                  <div className='flex w-full gap-4'>
                    <Input
                      className='w-full md:w-6/12'
                      size='sm'
                      type="text"
                      label="From Language"
                      placeholder="Auto"
                      autoComplete='on'
                      value={fromLanguage}
                      onValueChange={setFromLanguage}
                    />
                    <Input
                      className='w-full md:w-6/12'
                      size='sm'
                      type="text"
                      label="To Language"
                      autoComplete='on'
                      value={toLanguage}
                      onValueChange={setToLanguage}
                    />
                  </div>

                  <div className='w-full'>
                    <Textarea
                      label="System Instruction"
                      minRows={2}
                      description={"Override preset system instruction"}
                      placeholder={`Translate ${fromLanguage ? fromLanguage + " " : ""}to ${toLanguage}`}
                      value={systemInstruction}
                      onValueChange={setSystemInstruction}
                    />
                  </div>

                  <div className='flex flex-wrap md:flex-nowrap w-full gap-4'>
                    <div className='w-full md:w-1/5 flex flex-col gap-1'>
                      <div className='flex gap-1 items-start'>
                        <Select
                          className='flex-1'
                          size='sm'
                          label="Model"
                          placeholder={modelsLoading ? "Loading…" : "Click refresh →"}
                          isRequired={isOllama}
                          isLoading={modelsLoading}
                          selectedKeys={model ? [model] : []}
                          onSelectionChange={(keys) => {
                            const selected = [...keys][0]
                            if (selected) setModelValue(selected)
                          }}
                        >
                          {availableModels.map((m) => (
                            <SelectItem key={m} value={m}>
                              {m}
                            </SelectItem>
                          ))}
                        </Select>
                        <Button
                          isIconOnly
                          size='sm'
                          variant='flat'
                          className='mt-1'
                          isLoading={modelsLoading}
                          onPress={() => fetchModels()}
                          title="Refresh model list"
                        >
                          {!modelsLoading && <RefreshIcon />}
                        </Button>
                      </div>
                      {modelsError && (
                        <p className="text-tiny text-danger">{modelsError}</p>
                      )}
                    </div>

                    <div className='w-full md:w-1/5 flex'>
                      <Switch
                        size='sm'
                        isSelected={useStructuredMode}
                        onValueChange={setUseStructuredMode}
                      >
                      </Switch>
                      <div className="flex flex-col place-content-center gap-1">
                        <p className="text-small">Use Structured Mode</p>
                      </div>
                    </div>

                    <div className='w-full md:w-1/5'>
                      <Slider
                        label="Temperature"
                        size="md"
                        hideThumb={true}
                        step={0.05}
                        maxValue={2}
                        minValue={0}
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e))}
                      />
                    </div>

                    <div className='w-full md:w-1/5'>
                      <Slider
                        label="Batch Sizes"
                        size="md"
                        step={10}
                        maxValue={200}
                        minValue={10}
                        value={batchSizes}
                        onChange={(e) => typeof e === "number" ? setBatchSizes([e]) : setBatchSizes(e)}
                      />
                    </div>

                    <div className='w-full md:w-1/5'>
                      <Input
                        size='sm'
                        type="number"
                        min="1"
                        label="Rate Limit"
                        value={rateLimit.toString()}
                        onValueChange={(value) => setRateLimitValue(value)}
                        autoComplete='on'
                        endContent={
                          <div className="pointer-events-none flex items-center">
                            <span className="text-default-400 text-small">RPM</span>
                          </div>
                        }
                      />
                    </div>
                  </div>
                </div>
              </CardBody>
            </Card>
          </div>
        </form>

        <div className='w-full justify-between md:justify-center flex flex-wrap gap-1 sm:gap-4 mt-auto sticky top-0 backdrop-blur px-4 pt-4'>
          <FileUploadButton label={"Import SRT"} onFileSelect={async (file) => {
            try {
              const text = await file.text()
              const parsed = subtitleParser.fromSrt(text)
              setSrtInputText(text)
              setInputs(parsed.map(x => x.text))
            } catch (error) {
              alert(error.message ?? error)
            }
          }} />
          {!translatorRunningState && (
            <Button type='submit' form="translator-config-form" color="primary" isDisabled={!canStart()}>
              Start
            </Button>
          )}

          {translatorRunningState && (
            <Button color="danger" onClick={() => stopGeneration()} isLoading={!streamOutput}>
              Stop
            </Button>
          )}

          <Button color="primary" onClick={() => {
            downloadString(srtOutputText, "text/plain", "export.srt")
          }}>
            Export SRT
          </Button>
          <Divider className='mt-3 sm:mt-0' />
        </div>

        <div className="lg:flex lg:gap-4 px-4 mt-4">
          <div className="lg:w-1/2">
            <SubtitleCard label={"Input"}>
              <ol className="py-2 list-decimal line-marker ">
                {inputs.map((line, i) => {
                  return (
                    <li key={i} className=''>
                      <div className='ml-4 truncate'>
                        {line}
                      </div>
                    </li>
                  )
                })}
              </ol>
            </SubtitleCard>
          </div>

          <div className="lg:w-1/2">
            <SubtitleCard label={"Output"}>
              <ol className="py-2 list-decimal line-marker ">
                {outputs.map((line, i) => {
                  return (
                    <li key={i} className=''>
                      <div className='ml-4 truncate'>
                        {line}
                      </div>
                    </li>
                  )
                })}
                <pre className='px-2 text-wrap'>
                  {streamOutput}
                </pre>
              </ol>
            </SubtitleCard>

            {usageInformation && (
              <Card shadow="sm" className='mt-4 p-4'>
                <span><b>Estimated Usage</b></span>
                <span>Tokens: {usageInformation?.promptTokensUsed} + {usageInformation?.completionTokensUsed} = {usageInformation?.usedTokens}</span>
                {usageInformation?.wastedTokens > 0 && (
                  <span className={'text-danger'}>Wasted: {usageInformation?.promptTokensWasted} + {usageInformation?.completionTokensWasted} = {usageInformation?.wastedTokens} {usageInformation?.wastedPercent}</span>
                )}
                {usageInformation?.cachedTokens > 0 && (
                  <span className={'text-success'}>Cached: {usageInformation?.cachedTokens}</span>
                )}
                {usageInformation?.contextTokens > 0 && (
                  <span>Context: {usageInformation?.contextPromptTokens} + {usageInformation?.contextCompletionTokens} = {usageInformation?.contextTokens}</span>
                )}
                <span>{usageInformation?.promptRate} + {usageInformation?.completionRate} = {usageInformation?.rate} TPM {RPMInfomation} RPM</span>
              </Card>
            )}

          </div>
        </div>
      </div>
    </>
  )
}
