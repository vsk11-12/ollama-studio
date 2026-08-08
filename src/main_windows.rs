#![windows_subsystem = "windows"]

use base64::Engine;
use directories::ProjectDirs;
use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const MAX_HISTORY_MESSAGES: usize = 20;

// ---------------------------------------------------------------------------
// Config & Data Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SelectedFont {
    Proportional,
    Monospace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveView {
    Chat,
    Stats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Appearance,
    Parameters,
    Connection,
}

#[derive(Clone, Serialize, Deserialize)]
struct AppSettings {
    ollama_url: String,
    dark_mode: bool,
    zoom_factor: f32,
    selected_font: SelectedFont,
    base_font_size: f32,
    default_system_prompt: String,
    default_temperature: f32,
    default_top_p: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ollama_url: DEFAULT_OLLAMA_URL.to_string(),
            dark_mode: true,
            zoom_factor: 1.0,
            selected_font: SelectedFont::Proportional,
            base_font_size: 14.0,
            default_system_prompt: String::new(),
            default_temperature: 0.7,
            default_top_p: 0.9,
        }
    }
}

#[derive(Default, Debug, Clone)]
struct SessionStats {
    total_prompt_tokens: usize,
    total_completion_tokens: usize,
    running_models: Vec<RunningModel>,
}

#[derive(Deserialize, Debug, Clone)]
struct RunningModel {
    name: String,
    size: u64,
    size_vram: u64,
}

#[derive(Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
    #[serde(default)]
    prompt_tokens: Option<usize>,
    #[serde(default)]
    completion_tokens: Option<usize>,
    #[serde(default)]
    eval_duration_secs: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ChatTab {
    id: usize,
    title: String,
    messages: Vec<Message>,
    selected_model: String,
    #[serde(default)]
    system_prompt: String,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_temp() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}

enum AppEvent {
    ModelsFetched(Vec<String>),
    RunningModelsFetched(Vec<RunningModel>),
    FilePicked(Option<String>),
    StreamChunk {
        tab_id: usize,
        chunk: String,
    },
    StreamFinished {
        tab_id: usize,
        prompt_tokens: Option<usize>,
        completion_tokens: Option<usize>,
        eval_duration_secs: Option<f64>,
    },
    StreamError {
        tab_id: usize,
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MessageAction {
    None,
    StartEdit(usize, String),
    SaveEdit(usize, String),
    CancelEdit,
    Regenerate(usize),
}

// ---------------------------------------------------------------------------
// Modern UI Theme Tokens
// ---------------------------------------------------------------------------

struct ThemeTokens {
    bg_base: egui::Color32,
    bg_surface: egui::Color32,
    bg_subtle: egui::Color32,
    border_subtle: egui::Color32,
    border_strong: egui::Color32,
    accent_primary: egui::Color32,
    text_primary: egui::Color32,
    text_secondary: egui::Color32,
    user_bubble_bg: egui::Color32,
    assistant_bubble_bg: egui::Color32,
}

impl ThemeTokens {
    fn dark() -> Self {
        Self {
            bg_base: egui::Color32::from_rgb(18, 18, 20),
            bg_surface: egui::Color32::from_rgb(26, 26, 30),
            bg_subtle: egui::Color32::from_rgb(34, 34, 40),
            border_subtle: egui::Color32::from_rgb(44, 44, 52),
            border_strong: egui::Color32::from_rgb(64, 64, 76),
            accent_primary: egui::Color32::from_rgb(234, 88, 12),
            text_primary: egui::Color32::from_rgb(244, 244, 245),
            text_secondary: egui::Color32::from_rgb(161, 161, 170),
            user_bubble_bg: egui::Color32::from_rgb(34, 34, 42),
            assistant_bubble_bg: egui::Color32::TRANSPARENT,
        }
    }

    fn light() -> Self {
        Self {
            bg_base: egui::Color32::from_rgb(248, 249, 250),
            bg_surface: egui::Color32::from_rgb(255, 255, 255),
            bg_subtle: egui::Color32::from_rgb(241, 243, 245),
            border_subtle: egui::Color32::from_rgb(226, 232, 240),
            border_strong: egui::Color32::from_rgb(203, 213, 225),
            accent_primary: egui::Color32::from_rgb(234, 88, 12),
            text_primary: egui::Color32::from_rgb(15, 23, 42),
            text_secondary: egui::Color32::from_rgb(100, 116, 139),
            user_bubble_bg: egui::Color32::from_rgb(241, 245, 249),
            assistant_bubble_bg: egui::Color32::TRANSPARENT,
        }
    }
}

// ---------------------------------------------------------------------------
// Main State Struct
// ---------------------------------------------------------------------------

struct OllamaApp {
    // Shared Network Client & Sender
    http_client: Arc<reqwest::blocking::Client>,
    tx: Sender<AppEvent>,
    rx: Receiver<AppEvent>,

    // Sub-states
    settings: AppSettings,
    stats: SessionStats,

    // Chat Management
    tabs: Vec<ChatTab>,
    active_tab_id: usize,
    next_tab_id: usize,
    available_models: Vec<String>,

    // Layout & UI State
    input_text: String,
    selected_file: Option<String>,
    scroll_to_bottom: bool,
    sidebar_collapsed: bool,
    show_tab_config: bool,
    editing_tab_id: Option<usize>,
    rename_input: String,
    active_view: ActiveView,

    // Modal & Settings UI State
    show_settings: bool,
    active_settings_tab: SettingsTab,
    theme_dirty: bool,

    // Performance & Storage debouncing
    is_dirty: bool,
    last_save_time: Instant,
    toast_notification: Option<(String, Instant)>,

    // Generation state
    markdown_cache: CommonMarkCache,
    editing_msg: Option<(usize, usize, String)>,
    cancel_flag: Option<Arc<AtomicBool>>,
    is_generating: bool,
}

impl OllamaApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();

        let http_client = Arc::new(
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .unwrap_or_default(),
        );

        let settings = AppSettings::default();

        let tx_clone = tx.clone();
        let client_clone = http_client.clone();
        let ctx = cc.egui_ctx.clone();
        let url_clone = settings.ollama_url.clone();

        thread::spawn(move || {
            let models = fetch_ollama_models(&client_clone, &url_clone);
            let _ = tx_clone.send(AppEvent::ModelsFetched(models));
            ctx.request_repaint();
        });

        let initial_tabs = load_chats();
        let active_id = initial_tabs.first().map(|t| t.id).unwrap_or(0);
        let max_id = initial_tabs.iter().map(|t| t.id).max().unwrap_or(0);

        Self {
            http_client,
            tx,
            rx,
            settings,
            stats: SessionStats::default(),
            tabs: initial_tabs,
            active_tab_id: active_id,
            next_tab_id: max_id + 1,
            available_models: vec!["llama3.2-vision:latest".to_string(), "llama3.2:1b".to_string()],
            input_text: String::new(),
            selected_file: None,
            scroll_to_bottom: false,
            sidebar_collapsed: false,
            show_tab_config: false,
            editing_tab_id: None,
            rename_input: String::new(),
            active_view: ActiveView::Chat,
            show_settings: false,
            active_settings_tab: SettingsTab::Appearance,
            theme_dirty: true,
            is_dirty: false,
            last_save_time: Instant::now(),
            toast_notification: None,
            markdown_cache: CommonMarkCache::default(),
            editing_msg: None,
            cancel_flag: None,
            is_generating: false,
        }
    }

    fn show_toast(&mut self, message: &str, duration_secs: u64) {
        self.toast_notification = Some((
            message.to_string(),
            Instant::now() + Duration::from_secs(duration_secs),
        ));
    }

    fn mark_dirty(&mut self) {
        self.is_dirty = true;
    }

    fn create_new_chat(&mut self) {
        let new_id = self.next_tab_id;
        self.next_tab_id += 1;
        let fallback_model = self
            .available_models
            .first()
            .cloned()
            .unwrap_or_else(|| "llama3.2:1b".to_string());

        self.tabs.push(ChatTab {
            id: new_id,
            title: format!("Chat {}", new_id + 1),
            messages: Vec::new(),
            selected_model: fallback_model,
            system_prompt: self.settings.default_system_prompt.clone(),
            temperature: self.settings.default_temperature,
            top_p: self.settings.default_top_p,
        });

        self.switch_active_tab(new_id);
        self.mark_dirty();
        self.show_toast("Created new chat", 2);
    }

    fn switch_active_tab(&mut self, target_id: usize) {
        if self.active_tab_id == target_id {
            return;
        }

        if self.is_generating {
            self.stop_generation();
        }

        if let Some(idx) = self.get_active_tab_index() {
            let model = self.tabs[idx].selected_model.clone();
            let url = self.settings.ollama_url.clone();
            let client = self.http_client.clone();
            thread::spawn(move || {
                unload_ollama_model(&client, &url, &model);
            });
        }

        self.active_tab_id = target_id;
        self.scroll_to_bottom = true;
    }

    fn stop_generation(&mut self) {
        if let Some(flag) = &self.cancel_flag {
            flag.store(true, Ordering::Relaxed);
        }

        if let Some(idx) = self.get_active_tab_index() {
            let model = self.tabs[idx].selected_model.clone();
            let url = self.settings.ollama_url.clone();
            let client = self.http_client.clone();
            thread::spawn(move || {
                unload_ollama_model(&client, &url, &model);
            });
        }

        self.is_generating = false;
    }

    fn start_generation(
        &mut self,
        ctx: &egui::Context,
        active_idx: usize,
        attached_file: Option<String>,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag = Some(cancel.clone());
        self.is_generating = true;
        self.scroll_to_bottom = true;

        let tab = &self.tabs[active_idx];
        let model_name = tab.selected_model.clone();
        let system_prompt = tab.system_prompt.clone();
        let temperature = tab.temperature;
        let top_p = tab.top_p;
        let tab_id = tab.id;
        let url = self.settings.ollama_url.clone();

        let mut history = tab.messages.clone();
        if let Some(last) = history.last() {
            if last.role == "assistant" && last.content.is_empty() {
                history.pop();
            }
        }

        self.mark_dirty();

        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();
        let client = self.http_client.clone();

        thread::spawn(move || {
            query_ollama_stream(
                client,
                &url,
                &model_name,
                system_prompt,
                temperature,
                top_p,
                history,
                attached_file,
                tab_id,
                &tx,
                &ctx_clone,
                cancel,
            );
        });
    }

    fn fetch_running_models(&mut self, ctx: &egui::Context) {
        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();
        let url = self.settings.ollama_url.clone();
        let client = self.http_client.clone();

        thread::spawn(move || {
            let running = fetch_ollama_ps(&client, &url);
            let _ = tx.send(AppEvent::RunningModelsFetched(running));
            ctx_clone.request_repaint();
        });
    }

    fn unload_all_models(&mut self, ctx: &egui::Context) {
        let url = self.settings.ollama_url.clone();
        let models = self.stats.running_models.clone();
        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();
        let client = self.http_client.clone();

        thread::spawn(move || {
            for m in models {
                unload_ollama_model(&client, &url, &m.name);
            }

            let running = fetch_ollama_ps(&client, &url);
            let _ = tx.send(AppEvent::RunningModelsFetched(running));
            ctx_clone.request_repaint();
        });
        self.show_toast("Unloaded all models from RAM/VRAM", 3);
    }

    fn apply_theme_and_scale(&mut self, ctx: &egui::Context) {
        ctx.set_zoom_factor(self.settings.zoom_factor);

        let font_family = match self.settings.selected_font {
            SelectedFont::Proportional => egui::FontFamily::Proportional,
            SelectedFont::Monospace => egui::FontFamily::Monospace,
        };

        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(10.0, 6.0);

        style.text_styles = [
            (
                egui::TextStyle::Heading,
                egui::FontId::new(self.settings.base_font_size + 4.0, font_family.clone()),
            ),
            (
                egui::TextStyle::Body,
                egui::FontId::new(self.settings.base_font_size, font_family.clone()),
            ),
            (
                egui::TextStyle::Monospace,
                egui::FontId::new(self.settings.base_font_size - 1.0, egui::FontFamily::Monospace),
            ),
            (
                egui::TextStyle::Button,
                egui::FontId::new(self.settings.base_font_size - 1.0, font_family.clone()),
            ),
            (
                egui::TextStyle::Small,
                egui::FontId::new(self.settings.base_font_size - 3.0, font_family),
            ),
        ]
        .into();
        ctx.set_style(style);

        let tokens = if self.settings.dark_mode {
            ThemeTokens::dark()
        } else {
            ThemeTokens::light()
        };

        let mut visuals = if self.settings.dark_mode {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        };

        visuals.panel_fill = tokens.bg_surface;
        visuals.window_fill = tokens.bg_surface;
        visuals.faint_bg_color = tokens.bg_subtle;
        visuals.extreme_bg_color = tokens.bg_base;
        visuals.code_bg_color = tokens.bg_subtle;

        visuals.widgets.noninteractive.bg_fill = tokens.bg_surface;
        visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, tokens.text_primary);
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, tokens.border_subtle);

        visuals.widgets.inactive.bg_fill = tokens.bg_subtle;
        visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, tokens.text_primary);
        visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);

        visuals.widgets.hovered.bg_fill = if self.settings.dark_mode {
            egui::Color32::from_rgb(45, 45, 55)
        } else {
            egui::Color32::from_rgb(230, 235, 240)
        };
        visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, tokens.text_primary);
        visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);

        visuals.widgets.active.bg_fill = tokens.accent_primary;
        visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
        visuals.widgets.active.rounding = egui::Rounding::same(8.0);

        visuals.selection.bg_fill = tokens.accent_primary;
        visuals.window_stroke = egui::Stroke::new(1.0, tokens.border_subtle);
        visuals.window_rounding = egui::Rounding::same(14.0);

        ctx.set_visuals(visuals);
    }

    fn handle_events(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                AppEvent::ModelsFetched(models) => {
                    if !models.is_empty() {
                        if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == self.active_tab_id) {
                            if !models.contains(&tab.selected_model) {
                                tab.selected_model = models[0].clone();
                            }
                        }
                        self.available_models = models;
                    }
                }
                AppEvent::RunningModelsFetched(models) => {
                    self.stats.running_models = models;
                }
                AppEvent::FilePicked(file_path) => {
                    self.selected_file = file_path;
                }
                AppEvent::StreamChunk { tab_id, chunk } => {
                    if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) {
                        if let Some(last_msg) = tab.messages.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.content.push_str(&chunk);
                            }
                        }
                    }
                }
                AppEvent::StreamFinished {
                    tab_id,
                    prompt_tokens,
                    completion_tokens,
                    eval_duration_secs,
                } => {
                    if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) {
                        if let Some(last_msg) = tab.messages.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.prompt_tokens = prompt_tokens;
                                last_msg.completion_tokens = completion_tokens;
                                last_msg.eval_duration_secs = eval_duration_secs;
                            }
                        }
                    }

                    if let Some(p) = prompt_tokens {
                        self.stats.total_prompt_tokens += p;
                    }
                    if let Some(c) = completion_tokens {
                        self.stats.total_completion_tokens += c;
                    }

                    self.is_generating = false;
                    self.mark_dirty();
                }
                AppEvent::StreamError { tab_id, error } => {
                    if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) {
                        if let Some(last_msg) = tab.messages.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.content = format!("⚠️ Error: {}", error);
                            }
                        }
                    }
                    self.is_generating = false;
                    self.show_toast(&format!("Error: {}", error), 5);
                    self.mark_dirty();
                }
            }
        }
    }

    fn get_active_tab_index(&self) -> Option<usize> {
        self.tabs.iter().position(|t| t.id == self.active_tab_id)
    }

    fn render_stats_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let tokens = if self.settings.dark_mode {
            ThemeTokens::dark()
        } else {
            ThemeTokens::light()
        };

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.heading(
                            egui::RichText::new("System Analytics")
                                .strong()
                                .size(20.0)
                                .color(tokens.text_primary),
                        );
                        ui.label(
                            egui::RichText::new(
                                "Monitor session throughput and VRAM allocation across active models.",
                            )
                            .size(12.5)
                            .color(tokens.text_secondary),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_sized(
                                [120.0, 32.0],
                                egui::Button::new("🔄 Refresh Stats")
                                    .fill(tokens.bg_subtle)
                                    .rounding(egui::Rounding::same(8.0)),
                            )
                            .clicked()
                        {
                            self.fetch_running_models(ctx);
                        }

                        if ui
                            .add_sized(
                                [160.0, 32.0],
                                egui::Button::new(
                                    egui::RichText::new("⏹ Stop & Unload All")
                                        .color(egui::Color32::from_rgb(239, 68, 68)),
                                )
                                .fill(tokens.bg_subtle)
                                .rounding(egui::Rounding::same(8.0)),
                            )
                            .clicked()
                        {
                            if self.is_generating {
                                self.stop_generation();
                            }
                            self.unload_all_models(ctx);
                        }
                    });
                });

                ui.add_space(16.0);

                // --- Token Metric Cards ---
                ui.horizontal_top(|ui| {
                    let card_width = ((ui.available_width() - 24.0) / 3.0).max(180.0);

                    render_metric_card(
                        ui,
                        "PROMPT TOKENS",
                        &format!("{}", self.stats.total_prompt_tokens),
                        card_width,
                        &tokens,
                    );
                    ui.add_space(12.0);
                    render_metric_card(
                        ui,
                        "COMPLETION TOKENS",
                        &format!("{}", self.stats.total_completion_tokens),
                        card_width,
                        &tokens,
                    );
                    ui.add_space(12.0);
                    render_metric_card(
                        ui,
                        "TOTAL TOKENS",
                        &format!(
                            "{}",
                            self.stats.total_prompt_tokens + self.stats.total_completion_tokens
                        ),
                        card_width,
                        &tokens,
                    );
                });

                ui.add_space(20.0);

                // --- VRAM & Loaded Models ---
                egui::Frame::none()
                    .fill(tokens.bg_surface)
                    .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                    .rounding(egui::Rounding::same(12.0))
                    .inner_margin(egui::Margin::same(16.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.heading(
                                egui::RichText::new("🧠 Loaded Models in VRAM")
                                    .size(15.0)
                                    .strong()
                                    .color(tokens.text_primary),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} Active",
                                        self.stats.running_models.len()
                                    ))
                                    .size(12.0)
                                    .color(tokens.text_secondary),
                                );
                            });
                        });
                        ui.add_space(12.0);

                        if self.stats.running_models.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(12.0);
                                ui.label(
                                    egui::RichText::new("No LLMs currently loaded in memory.")
                                        .italics()
                                        .size(13.0)
                                        .color(tokens.text_secondary),
                                );
                                ui.add_space(12.0);
                            });
                        } else {
                            egui::Grid::new("running_models_grid")
                                .striped(true)
                                .min_col_width(140.0)
                                .spacing([20.0, 10.0])
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new("Model Identifier")
                                            .strong()
                                            .color(tokens.text_secondary),
                                    );
                                    ui.label(
                                        egui::RichText::new("VRAM Usage")
                                            .strong()
                                            .color(tokens.text_secondary),
                                    );
                                    ui.label(
                                        egui::RichText::new("Total Footprint")
                                            .strong()
                                            .color(tokens.text_secondary),
                                    );
                                    ui.label(
                                        egui::RichText::new("VRAM Ratio")
                                            .strong()
                                            .color(tokens.text_secondary),
                                    );
                                    ui.end_row();

                                    for model in &self.stats.running_models {
                                        let vram_gb = model.size_vram as f64 / 1e9;
                                        let total_gb = model.size as f64 / 1e9;
                                        let ratio = if model.size > 0 {
                                            (model.size_vram as f32 / model.size as f32).clamp(0.0, 1.0)
                                        } else {
                                            0.0
                                        };

                                        ui.label(
                                            egui::RichText::new(&model.name)
                                                .strong()
                                                .color(tokens.text_primary),
                                        );
                                        ui.label(format!("{:.2} GB", vram_gb));
                                        ui.label(format!("{:.2} GB", total_gb));
                                        ui.add(
                                            egui::ProgressBar::new(ratio)
                                                .text(format!("{:.0}%", ratio * 100.0))
                                                .desired_width(120.0),
                                        );
                                        ui.end_row();
                                    }
                                });
                        }
                    });
            });
    }
}

fn render_metric_card(
    ui: &mut egui::Ui,
    label: &str,
    value: &str,
    width: f32,
    tokens: &ThemeTokens,
) {
    egui::Frame::none()
        .fill(tokens.bg_subtle)
        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
        .rounding(egui::Rounding::same(10.0))
        .inner_margin(egui::Margin::same(14.0))
        .show(ui, |ui| {
            ui.set_width(width);
            ui.label(
                egui::RichText::new(label)
                    .size(10.5)
                    .strong()
                    .color(tokens.text_secondary),
            );
            ui.add_space(4.0);
            ui.heading(
                egui::RichText::new(value)
                    .size(22.0)
                    .strong()
                    .color(tokens.accent_primary),
            );
        });
}

// ---------------------------------------------------------------------------
// eframe Implementation
// ---------------------------------------------------------------------------

impl eframe::App for OllamaApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        save_chats_sync(&self.tabs);
        if let Some(idx) = self.get_active_tab_index() {
            let url = self.settings.ollama_url.clone();
            let model = self.tabs[idx].selected_model.clone();
            let client = self.http_client.clone();
            thread::spawn(move || {
                unload_ollama_model(&client, &url, &model);
            });
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Keyboard Shortcuts ---
        ctx.input(|i| {
            if i.modifiers.command && i.key_pressed(egui::Key::N) {
                self.create_new_chat();
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Comma) {
                self.show_settings = !self.show_settings;
            }
        });

        // --- Debounced Storage Auto-Save ---
        if self.is_dirty && self.last_save_time.elapsed() >= Duration::from_secs(2) {
            save_chats_async(self.tabs.clone());
            self.is_dirty = false;
            self.last_save_time = Instant::now();
        }

        self.handle_events();

        if self.theme_dirty {
            self.apply_theme_and_scale(ctx);
            self.theme_dirty = false;
        }

        let tokens = if self.settings.dark_mode {
            ThemeTokens::dark()
        } else {
            ThemeTokens::light()
        };

        // --- Sidebar Panel ---
        if !self.sidebar_collapsed {
            egui::SidePanel::left("sidebar")
                .default_width(240.0)
                .width_range(200.0..=320.0)
                .frame(
                    egui::Frame::side_top_panel(ctx.style().as_ref())
                        .fill(tokens.bg_surface)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .inner_margin(egui::Margin::same(12.0)),
                )
                .show(ctx, |ui| {
                    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                        ui.add_space(4.0);
                        let settings_btn = ui.add_sized(
                            [ui.available_width(), 32.0],
                            egui::Button::new(
                                egui::RichText::new("⚙ Settings")
                                    .size(13.0)
                                    .color(tokens.text_primary),
                            )
                            .fill(tokens.bg_subtle)
                            .rounding(egui::Rounding::same(8.0)),
                        );

                        if settings_btn.clicked() {
                            self.show_settings = !self.show_settings;
                        }

                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);

                        ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Ollama Studio")
                                        .strong()
                                        .size(15.0)
                                        .color(tokens.text_primary),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.button("⏴").on_hover_text("Hide Sidebar").clicked() {
                                            self.sidebar_collapsed = true;
                                        }
                                    },
                                );
                            });

                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui
                                    .selectable_label(self.active_view == ActiveView::Chat, "💬 Chats")
                                    .clicked()
                                {
                                    self.active_view = ActiveView::Chat;
                                }
                                if ui
                                    .selectable_label(self.active_view == ActiveView::Stats, "📊 Stats")
                                    .clicked()
                                {
                                    self.active_view = ActiveView::Stats;
                                    self.fetch_running_models(ctx);
                                }
                            });

                            ui.add_space(10.0);

                            if self.active_view == ActiveView::Chat {
                                let new_chat_btn = ui.add_sized(
                                    [ui.available_width(), 34.0],
                                    egui::Button::new(
                                        egui::RichText::new("+ New Chat")
                                            .strong()
                                            .size(12.5)
                                            .color(egui::Color32::WHITE),
                                    )
                                    .fill(tokens.accent_primary)
                                    .rounding(egui::Rounding::same(8.0)),
                                );

                                if new_chat_btn.clicked() {
                                    self.create_new_chat();
                                }

                                ui.add_space(14.0);
                                ui.label(
                                    egui::RichText::new("CONVERSATIONS")
                                        .size(10.0)
                                        .strong()
                                        .color(tokens.text_secondary),
                                );
                                ui.add_space(6.0);

                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let tabs_len = self.tabs.len();
                                    let mut tab_to_close = None;
                                    let mut tab_to_edit = None;
                                    let mut tab_to_save_rename = None;
                                    let mut tab_to_switch = None;

                                    for tab in &mut self.tabs {
                                        let is_active = tab.id == self.active_tab_id;

                                        ui.horizontal(|ui| {
                                            if Some(tab.id) == self.editing_tab_id {
                                                let text_edit = ui.add(
                                                    egui::TextEdit::singleline(
                                                        &mut self.rename_input,
                                                    )
                                                    .desired_width(120.0),
                                                );
                                                if text_edit.lost_focus()
                                                    || ui.input(|i| i.key_pressed(egui::Key::Enter))
                                                {
                                                    tab_to_save_rename = Some(tab.id);
                                                }
                                            } else {
                                                let tab_bg = if is_active {
                                                    tokens.bg_subtle
                                                } else {
                                                    egui::Color32::TRANSPARENT
                                                };

                                                egui::Frame::none()
                                                    .fill(tab_bg)
                                                    .rounding(egui::Rounding::same(6.0))
                                                    .inner_margin(egui::Margin::symmetric(
                                                        8.0, 6.0,
                                                    ))
                                                    .show(ui, |ui| {
                                                        ui.set_width(ui.available_width());
                                                        ui.horizontal(|ui| {
                                                            let title_color = if is_active {
                                                                tokens.text_primary
                                                            } else {
                                                                tokens.text_secondary
                                                            };

                                                            let title_btn = ui.add(
                                                                egui::Label::new(
                                                                    egui::RichText::new(&tab.title)
                                                                        .size(13.0)
                                                                        .color(title_color),
                                                                )
                                                                .sense(egui::Sense::click()),
                                                            );

                                                            if title_btn.clicked() {
                                                                tab_to_switch = Some(tab.id);
                                                            }

                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    if tabs_len > 1
                                                                        && ui
                                                                            .small_button("×")
                                                                            .clicked()
                                                                    {
                                                                        tab_to_close =
                                                                            Some(tab.id);
                                                                    }
                                                                    if ui
                                                                        .small_button("✏")
                                                                        .clicked()
                                                                    {
                                                                        tab_to_edit = Some((
                                                                            tab.id,
                                                                            tab.title.clone(),
                                                                        ));
                                                                    }
                                                                },
                                                            );
                                                        });
                                                    });
                                            }
                                        });
                                        ui.add_space(2.0);
                                    }

                                    if let Some(id) = tab_to_switch {
                                        self.switch_active_tab(id);
                                    }

                                    if let Some((id, title)) = tab_to_edit {
                                        self.editing_tab_id = Some(id);
                                        self.rename_input = title;
                                    }

                                    if let Some(id) = tab_to_save_rename {
                                        if let Some(tab) =
                                            self.tabs.iter_mut().find(|t| t.id == id)
                                        {
                                            if !self.rename_input.trim().is_empty() {
                                                tab.title = self.rename_input.clone();
                                            }
                                        }
                                        self.editing_tab_id = None;
                                        self.rename_input.clear();
                                        self.mark_dirty();
                                    }

                                    if let Some(id) = tab_to_close {
                                        self.tabs.retain(|t| t.id != id);
                                        if self.active_tab_id == id {
                                            let first_id = self.tabs.first().map(|t| t.id);
                                            if let Some(target_id) = first_id {
                                                self.switch_active_tab(target_id);
                                            }
                                        }
                                        self.mark_dirty();
                                    }
                                });
                            }
                        });
                    });
                });
        }

        // --- Main Surface Panel ---
        egui::CentralPanel::default()
            .frame(
                egui::Frame::central_panel(ctx.style().as_ref())
                    .fill(tokens.bg_base)
                    .inner_margin(egui::Margin::symmetric(24.0, 12.0)),
            )
            .show(ctx, |ui| {
                if self.active_view == ActiveView::Stats {
                    self.render_stats_tab(ui, ctx);
                    return;
                }

                // Top Bar Control Panel
                ui.horizontal(|ui| {
                    if self.sidebar_collapsed {
                        if ui.button("⏸").on_hover_text("Open Sidebar").clicked() {
                            self.sidebar_collapsed = false;
                        }
                        ui.add_space(8.0);
                    }

                    if let Some(idx) = self.get_active_tab_index() {
                        let selected_text = self.tabs[idx].selected_model.clone();
                        let mut selected_model_changed = false;

                        egui::ComboBox::from_id_salt("model_selector")
                            .selected_text(
                                egui::RichText::new(format!("🤖 {}", selected_text))
                                    .strong()
                                    .color(tokens.text_primary),
                            )
                            .show_ui(ui, |ui: &mut egui::Ui| {
                                for model in &self.available_models {
                                    if ui
                                        .selectable_value(
                                            &mut self.tabs[idx].selected_model,
                                            model.clone(),
                                            model,
                                        )
                                        .changed()
                                    {
                                        selected_model_changed = true;
                                    }
                                }
                            });

                        if selected_model_changed {
                            let old_model = selected_text;
                            let url = self.settings.ollama_url.clone();
                            let client = self.http_client.clone();
                            self.mark_dirty();
                            thread::spawn(move || {
                                unload_ollama_model(&client, &url, &old_model);
                            });
                        }

                        if ui
                            .button(if self.show_tab_config {
                                "⚙ Options ▲"
                            } else {
                                "⚙ Options ▼"
                            })
                            .on_hover_text("System Prompt & Hyperparameters")
                            .clicked()
                        {
                            self.show_tab_config = !self.show_tab_config;
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("🔄 Refresh").clicked() {
                            let tx_clone = self.tx.clone();
                            let ctx_clone = ctx.clone();
                            let url = self.settings.ollama_url.clone();
                            let client = self.http_client.clone();
                            thread::spawn(move || {
                                let models = fetch_ollama_models(&client, &url);
                                let _ = tx_clone.send(AppEvent::ModelsFetched(models));
                                ctx_clone.request_repaint();
                            });
                        }

                        if ui.button("🗑 Clear").clicked() {
                            if self.is_generating {
                                self.stop_generation();
                            }
                            if let Some(idx) = self.get_active_tab_index() {
                                self.tabs[idx].messages.clear();
                                self.mark_dirty();
                            }
                        }
                    });
                });

                // Tab Specific Hyperparameters Configuration Drawer
                if self.show_tab_config {
                    if let Some(idx) = self.get_active_tab_index() {
                        ui.add_space(8.0);
                        egui::Frame::none()
                            .fill(tokens.bg_surface)
                            .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                            .rounding(egui::Rounding::same(10.0))
                            .inner_margin(egui::Margin::same(12.0))
                            .show(ui, |ui| {
                                ui.vertical(|ui| {
                                    ui.label(
                                        egui::RichText::new("SYSTEM PROMPT")
                                            .size(10.0)
                                            .strong()
                                            .color(tokens.text_secondary),
                                    );
                                    if ui
                                        .add(
                                            egui::TextEdit::singleline(
                                                &mut self.tabs[idx].system_prompt,
                                            )
                                            .desired_width(f32::INFINITY)
                                            .hint_text("Optional system persona or custom instructions..."),
                                        )
                                        .changed()
                                    {
                                        self.mark_dirty();
                                    }

                                    ui.add_space(6.0);
                                    ui.horizontal(|ui| {
                                        ui.label("Temperature:");
                                        if ui
                                            .add(egui::Slider::new(
                                                &mut self.tabs[idx].temperature,
                                                0.0..=1.5,
                                            ))
                                            .changed()
                                        {
                                            self.mark_dirty();
                                        }

                                        ui.add_space(16.0);

                                        ui.label("Top P:");
                                        if ui
                                            .add(egui::Slider::new(
                                                &mut self.tabs[idx].top_p,
                                                0.0..=1.0,
                                            ))
                                            .changed()
                                        {
                                            self.mark_dirty();
                                        }
                                    });
                                });
                            });
                    }
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                let active_idx = match self.get_active_tab_index() {
                    Some(idx) => idx,
                    None => return,
                };

                let reserved_bottom_space = 170.0;
                let available_height = (ui.available_height() - reserved_bottom_space).max(100.0);
                let total_avail_width = ui.available_width();
                let conversation_width = total_avail_width.min(768.0);
                let side_margin = ((total_avail_width - conversation_width) / 2.0).max(0.0);

                let mut message_action = MessageAction::None;
                let mut is_scrolled_up = false;

                // --- Scrollable Chat Viewport ---
                let scroll_output = egui::ScrollArea::vertical()
                    .max_height(available_height)
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if side_margin > 0.0 {
                                ui.add_space(side_margin);
                            }

                            ui.vertical(|ui| {
                                ui.set_width(conversation_width);

                                if self.tabs[active_idx].messages.is_empty() {
                                    ui.add_space(60.0);
                                    ui.vertical_centered(|ui| {
                                        ui.heading(
                                            egui::RichText::new("How can I help you today?")
                                                .size(24.0)
                                                .strong()
                                                .color(tokens.text_primary),
                                        );
                                        ui.add_space(8.0);
                                        ui.label(
                                            egui::RichText::new(
                                                "Ask a question, upload a document, or generate content.",
                                            )
                                            .size(13.0)
                                            .color(tokens.text_secondary),
                                        );
                                    });
                                }

                                let msg_count = self.tabs[active_idx].messages.len();

                                for msg_idx in 0..msg_count {
                                    let msg = &self.tabs[active_idx].messages[msg_idx];
                                    let is_user = msg.role == "user";

                                    let is_editing_this = match &self.editing_msg {
                                        Some((t_id, idx, _)) => {
                                            *t_id == self.active_tab_id && *idx == msg_idx
                                        }
                                        None => false,
                                    };

                                    if is_editing_this {
                                        let (_, _, ref mut edit_text) =
                                            self.editing_msg.as_mut().unwrap();

                                        egui::Frame::none()
                                            .fill(tokens.bg_subtle)
                                            .stroke(egui::Stroke::new(
                                                1.0,
                                                tokens.border_strong,
                                            ))
                                            .rounding(egui::Rounding::same(12.0))
                                            .inner_margin(egui::Margin::same(14.0))
                                            .show(ui, |ui| {
                                                ui.set_width(ui.available_width());
                                                ui.vertical(|ui| {
                                                    ui.label(
                                                        egui::RichText::new("Edit Query")
                                                            .strong()
                                                            .size(12.0)
                                                            .color(tokens.text_secondary),
                                                    );
                                                    ui.add_space(4.0);

                                                    egui::ScrollArea::vertical()
                                                        .max_height(120.0)
                                                        .show(ui, |ui| {
                                                            ui.add(
                                                                egui::TextEdit::multiline(edit_text)
                                                                    .desired_width(f32::INFINITY),
                                                            );
                                                        });

                                                    ui.add_space(8.0);
                                                    ui.horizontal(|ui| {
                                                        if ui
                                                            .button("Save & Resend")
                                                            .clicked()
                                                        {
                                                            message_action =
                                                                MessageAction::SaveEdit(
                                                                    msg_idx,
                                                                    edit_text.clone(),
                                                                );
                                                        }
                                                        if ui.button("Cancel").clicked() {
                                                            message_action =
                                                                MessageAction::CancelEdit;
                                                        }
                                                    });
                                                });
                                            });
                                    } else {
                                        let is_last_assistant = !is_user && msg_idx == msg_count - 1;

                                        let action = render_claude_message(
                                            ui,
                                            &mut self.markdown_cache,
                                            msg,
                                            is_user,
                                            is_last_assistant,
                                            self.is_generating,
                                            msg_idx,
                                            &tokens,
                                        );

                                        if action != MessageAction::None {
                                            message_action = action;
                                        }
                                    }
                                    ui.add_space(14.0);
                                }

                                // Pulsing Thinking Indicator during streams
                                if self.is_generating {
                                    let pulse = ctx.animate_bool_with_time(
                                        egui::Id::new("thinking_pulse"),
                                        true,
                                        0.6,
                                    );
                                    let alpha = ((pulse * std::f32::consts::TAU).sin() * 0.35 + 0.65)
                                        .clamp(0.2, 1.0);
                                    let animated_color = tokens.accent_primary.linear_multiply(alpha);

                                    ui.horizontal(|ui| {
                                        ui.add(egui::Spinner::new());
                                        ui.add_space(6.0);
                                        ui.label(
                                            egui::RichText::new("Generating response...")
                                                .italics()
                                                .strong()
                                                .color(animated_color),
                                        );
                                        ui.add_space(12.0);
                                        if ui.button("⏹ Stop").clicked() {
                                            self.stop_generation();
                                        }
                                    });
                                    ui.add_space(12.0);
                                }

                                if self.scroll_to_bottom {
                                    ui.scroll_to_cursor(Some(egui::Align::Max));
                                    self.scroll_to_bottom = false;
                                }
                            });
                        });
                    });

                let max_offset_y = scroll_output.content_size.y - scroll_output.inner_rect.height();
                if max_offset_y > 40.0 && (max_offset_y - scroll_output.state.offset.y) > 40.0 {
                    is_scrolled_up = true;
                }

                match message_action {
                    MessageAction::StartEdit(idx, text) => {
                        self.editing_msg = Some((self.active_tab_id, idx, text));
                    }
                    MessageAction::CancelEdit => {
                        self.editing_msg = None;
                    }
                    MessageAction::SaveEdit(idx, updated_prompt) => {
                        if self.is_generating {
                            self.stop_generation();
                        }
                        self.editing_msg = None;

                        if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == self.active_tab_id)
                        {
                            tab.messages[idx].content = updated_prompt;
                            tab.messages.truncate(idx + 1);
                            tab.messages.push(Message {
                                role: "assistant".to_string(),
                                content: String::new(),
                                prompt_tokens: None,
                                completion_tokens: None,
                                eval_duration_secs: None,
                            });
                        }
                        self.start_generation(ctx, active_idx, None);
                    }
                    MessageAction::Regenerate(idx) => {
                        if self.is_generating {
                            self.stop_generation();
                        }
                        if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == self.active_tab_id)
                        {
                            if idx < tab.messages.len() {
                                tab.messages.truncate(idx);
                                tab.messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: String::new(),
                                    prompt_tokens: None,
                                    completion_tokens: None,
                                    eval_duration_secs: None,
                                });
                            }
                        }
                        self.start_generation(ctx, active_idx, None);
                    }
                    MessageAction::None => {}
                }

                ui.add_space(4.0);

                if is_scrolled_up {
                    ui.horizontal(|ui| {
                        ui.vertical_centered(|ui| {
                            let jump_btn = ui.add(
                                egui::Button::new(
                                    egui::RichText::new("⬇ Scroll to bottom")
                                        .size(11.5)
                                        .strong()
                                        .color(tokens.text_primary),
                                )
                                .fill(tokens.bg_subtle)
                                .stroke(egui::Stroke::new(1.0, tokens.border_strong))
                                .rounding(egui::Rounding::same(12.0)),
                            );

                            if jump_btn.clicked() {
                                self.scroll_to_bottom = true;
                            }
                        });
                    });
                    ui.add_space(4.0);
                } else {
                    ui.add_space(2.0);
                }

                // --- Modern Centered Input Console ---
                ui.horizontal(|ui| {
                    if side_margin > 0.0 {
                        ui.add_space(side_margin);
                    }

                    egui::Frame::none()
                        .fill(tokens.bg_surface)
                        .stroke(egui::Stroke::new(1.0, tokens.border_strong))
                        .rounding(egui::Rounding::same(16.0))
                        .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                        .show(ui, |ui| {
                            ui.set_width(conversation_width);
                            ui.vertical(|ui| {
                                let mut clear_selected_file = false;

                                if let Some(file_path) = &self.selected_file {
                                    let path = std::path::Path::new(file_path);
                                    let filename = path
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("Attachment");
                                    let ext = path
                                        .extension()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("")
                                        .to_lowercase();
                                    let is_image = [
                                        "png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif", "svg",
                                    ]
                                    .contains(&ext.as_str());

                                    let (icon_symbol, type_badge) = if is_image {
                                        ("🖼", "IMAGE")
                                    } else {
                                        ("📄", "FILE")
                                    };

                                    egui::Frame::none()
                                        .fill(tokens.bg_subtle)
                                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                                        .rounding(egui::Rounding::same(8.0))
                                        .inner_margin(egui::Margin::symmetric(10.0, 6.0))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new(icon_symbol)
                                                        .size(15.0)
                                                        .color(tokens.accent_primary),
                                                );
                                                ui.add_space(2.0);
                                                ui.vertical(|ui| {
                                                    ui.label(
                                                        egui::RichText::new(filename)
                                                            .size(12.0)
                                                            .strong()
                                                            .color(tokens.text_primary),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(type_badge)
                                                            .size(9.0)
                                                            .strong()
                                                            .color(tokens.text_secondary),
                                                    );
                                                });
                                                ui.add_space(10.0);
                                                if ui
                                                    .small_button("✕")
                                                    .on_hover_text("Remove attachment")
                                                    .clicked()
                                                {
                                                    clear_selected_file = true;
                                                }
                                            });
                                        });

                                    ui.add_space(6.0);
                                }

                                if clear_selected_file {
                                    self.selected_file = None;
                                }

                                let mut enter_pressed = false;

                                egui::ScrollArea::vertical()
                                    .max_height(120.0)
                                    .show(ui, |ui| {
                                        let text_edit = ui.add(
                                            egui::TextEdit::multiline(&mut self.input_text)
                                                .hint_text(if self.is_generating {
                                                    "Awaiting Ollama response..."
                                                } else {
                                                    "Message Ollama Studio (Enter to send, Shift+Enter for newline)..."
                                                })
                                                .interactive(!self.is_generating)
                                                .desired_rows(2)
                                                .desired_width(f32::INFINITY)
                                                .frame(false),
                                        );

                                        enter_pressed = text_edit.has_focus()
                                            && ui.input(|i| {
                                                i.key_pressed(egui::Key::Enter)
                                                    && !i.modifiers.shift
                                            });
                                    });

                                ui.add_space(4.0);

                                ui.horizontal(|ui| {
                                    if ui
                                        .add_enabled(
                                            !self.is_generating,
                                            egui::Button::new("📎 Attach"),
                                        )
                                        .clicked()
                                    {
                                        let tx = self.tx.clone();
                                        let ctx_clone = ctx.clone();
                                        thread::spawn(move || {
                                            if let Some(path) = rfd::FileDialog::new().pick_file() {
                                                let _ = tx.send(AppEvent::FilePicked(Some(
                                                    path.display().to_string(),
                                                )));
                                                ctx_clone.request_repaint();
                                            }
                                        });
                                    }

                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if self.is_generating {
                                                if ui.button("⏹ Stop").clicked() {
                                                    self.stop_generation();
                                                }
                                            } else {
                                                let send_btn = ui.add_sized(
                                                    [64.0, 28.0],
                                                    egui::Button::new(
                                                        egui::RichText::new("Send ➔")
                                                            .strong()
                                                            .color(egui::Color32::WHITE),
                                                    )
                                                    .fill(tokens.accent_primary)
                                                    .rounding(egui::Rounding::same(8.0)),
                                                );

                                                if (send_btn.clicked() || enter_pressed)
                                                    && !self.input_text.trim().is_empty()
                                                {
                                                    let user_message =
                                                        self.input_text.trim().to_string();
                                                    let file_to_attach = self.selected_file.take();

                                                    self.tabs[active_idx].messages.push(Message {
                                                        role: "user".to_string(),
                                                        content: user_message,
                                                        prompt_tokens: None,
                                                        completion_tokens: None,
                                                        eval_duration_secs: None,
                                                    });

                                                    self.tabs[active_idx].messages.push(Message {
                                                        role: "assistant".to_string(),
                                                        content: String::new(),
                                                        prompt_tokens: None,
                                                        completion_tokens: None,
                                                        eval_duration_secs: None,
                                                    });

                                                    self.input_text.clear();
                                                    self.start_generation(
                                                        ctx,
                                                        active_idx,
                                                        file_to_attach,
                                                    );
                                                }
                                            }
                                        },
                                    );
                                });
                            });
                        });
                });
            });

        // Toast Notification Overlay
        if let Some((msg, expiry)) = &self.toast_notification {
            if Instant::now() < *expiry {
                egui::Area::new(egui::Id::new("toast_area"))
                    .anchor(egui::Align2::RIGHT_BOTTOM, [-20.0, -20.0])
                    .show(ctx, |ui| {
                        egui::Frame::none()
                            .fill(tokens.bg_surface)
                            .stroke(egui::Stroke::new(1.0, tokens.accent_primary))
                            .rounding(egui::Rounding::same(8.0))
                            .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(msg)
                                        .size(12.5)
                                        .strong()
                                        .color(tokens.text_primary),
                                );
                            });
                    });
            }
        }

        // Settings Dialog Window
        if self.show_settings {
            egui::Window::new("Settings & Preferences")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .frame(
                    egui::Frame::window(ctx.style().as_ref())
                        .fill(tokens.bg_surface)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(14.0))
                        .inner_margin(egui::Margin::same(20.0)),
                )
                .show(ctx, |ui| {
                    ui.set_width(420.0);

                    ui.horizontal(|ui| {
                        ui.heading(
                            egui::RichText::new("Settings")
                                .strong()
                                .size(18.0)
                                .color(tokens.text_primary),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("✕").clicked() {
                                self.show_settings = false;
                            }
                        });
                    });
                    ui.add_space(10.0);

                    ui.horizontal(|ui| {
                        let app_active = self.active_settings_tab == SettingsTab::Appearance;
                        let param_active = self.active_settings_tab == SettingsTab::Parameters;
                        let conn_active = self.active_settings_tab == SettingsTab::Connection;

                        if ui
                            .selectable_label(app_active, "🎨 Appearance")
                            .clicked()
                        {
                            self.active_settings_tab = SettingsTab::Appearance;
                        }
                        if ui
                            .selectable_label(param_active, "🎛 Parameters")
                            .clicked()
                        {
                            self.active_settings_tab = SettingsTab::Parameters;
                        }
                        if ui
                            .selectable_label(conn_active, "🔌 Connection")
                            .clicked()
                        {
                            self.active_settings_tab = SettingsTab::Connection;
                        }
                    });
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(12.0);

                    match self.active_settings_tab {
                        SettingsTab::Appearance => {
                            egui::Frame::none()
                                .fill(tokens.bg_subtle)
                                .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                                .rounding(egui::Rounding::same(8.0))
                                .inner_margin(egui::Margin::same(12.0))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(
                                        egui::RichText::new("THEME & SCALING")
                                            .strong()
                                            .size(11.0)
                                            .color(tokens.text_secondary),
                                    );
                                    ui.add_space(8.0);

                                    egui::Grid::new("settings_appearance_grid")
                                        .num_columns(2)
                                        .spacing([16.0, 10.0])
                                        .show(ui, |ui| {
                                            ui.label("Theme Mode:");
                                            ui.horizontal(|ui| {
                                                if ui
                                                    .selectable_value(&mut self.settings.dark_mode, true, "🌙 Dark")
                                                    .changed()
                                                {
                                                    self.theme_dirty = true;
                                                }
                                                if ui
                                                    .selectable_value(&mut self.settings.dark_mode, false, "☀️ Light")
                                                    .changed()
                                                {
                                                    self.theme_dirty = true;
                                                }
                                            });
                                            ui.end_row();

                                            ui.label("UI Scale:");
                                            ui.horizontal(|ui| {
                                                if ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut self.settings.zoom_factor,
                                                            0.75..=1.50,
                                                        )
                                                        .step_by(0.05)
                                                        .custom_formatter(|n, _| {
                                                            format!("{:.0}%", n * 100.0)
                                                        }),
                                                    )
                                                    .changed()
                                                {
                                                    self.theme_dirty = true;
                                                }
                                                if ui.small_button("Reset").clicked() {
                                                    self.settings.zoom_factor = 1.0;
                                                    self.theme_dirty = true;
                                                }
                                            });
                                            ui.end_row();
                                        });
                                });

                            ui.add_space(10.0);

                            egui::Frame::none()
                                .fill(tokens.bg_subtle)
                                .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                                .rounding(egui::Rounding::same(8.0))
                                .inner_margin(egui::Margin::same(12.0))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(
                                        egui::RichText::new("TYPOGRAPHY")
                                            .strong()
                                            .size(11.0)
                                            .color(tokens.text_secondary),
                                    );
                                    ui.add_space(8.0);

                                    egui::Grid::new("settings_typography_grid")
                                        .num_columns(2)
                                        .spacing([16.0, 10.0])
                                        .show(ui, |ui| {
                                            ui.label("Font Family:");
                                            let font_changed = egui::ComboBox::from_id_salt(
                                                "font_family_select",
                                            )
                                            .selected_text(match self.settings.selected_font {
                                                SelectedFont::Proportional => {
                                                    "Proportional (Sans)"
                                                }
                                                SelectedFont::Monospace => "Monospace (Code)",
                                            })
                                            .show_ui(ui, |ui: &mut egui::Ui| {
                                                let mut changed = false;
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.settings.selected_font,
                                                        SelectedFont::Proportional,
                                                        "Proportional (Sans)",
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.settings.selected_font,
                                                        SelectedFont::Monospace,
                                                        "Monospace (Code)",
                                                    )
                                                    .changed();
                                                changed
                                            })
                                            .inner
                                            .unwrap_or(false);

                                            if font_changed {
                                                self.theme_dirty = true;
                                            }
                                            ui.end_row();

                                            ui.label("Font Size:");
                                            ui.horizontal(|ui| {
                                                if ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut self.settings.base_font_size,
                                                            11.0..=20.0,
                                                        )
                                                        .suffix(" px"),
                                                    )
                                                    .changed()
                                                {
                                                    self.theme_dirty = true;
                                                }
                                                if ui.small_button("Reset").clicked() {
                                                    self.settings.base_font_size = 14.0;
                                                    self.theme_dirty = true;
                                                }
                                            });
                                            ui.end_row();
                                        });
                                });
                        }
                        SettingsTab::Parameters => {
                            egui::Frame::none()
                                .fill(tokens.bg_subtle)
                                .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                                .rounding(egui::Rounding::same(8.0))
                                .inner_margin(egui::Margin::same(12.0))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(
                                        egui::RichText::new("GLOBAL MODEL DEFAULTS")
                                            .strong()
                                            .size(11.0)
                                            .color(tokens.text_secondary),
                                    );
                                    ui.add_space(8.0);

                                    egui::Grid::new("settings_parameters_grid")
                                        .num_columns(2)
                                        .spacing([16.0, 10.0])
                                        .show(ui, |ui| {
                                            ui.label("Default Temp:");
                                            ui.add(egui::Slider::new(
                                                &mut self.settings.default_temperature,
                                                0.0..=1.5,
                                            ));
                                            ui.end_row();

                                            ui.label("Default Top P:");
                                            ui.add(egui::Slider::new(
                                                &mut self.settings.default_top_p,
                                                0.0..=1.0,
                                            ));
                                            ui.end_row();
                                        });

                                    ui.add_space(10.0);
                                    ui.label("Default System Prompt:");
                                    ui.add(
                                        egui::TextEdit::multiline(&mut self.settings.default_system_prompt)
                                            .desired_width(f32::INFINITY)
                                            .desired_rows(3)
                                            .hint_text("Default instructions for newly generated tabs..."),
                                    );
                                });
                        }
                        SettingsTab::Connection => {
                            egui::Frame::none()
                                .fill(tokens.bg_subtle)
                                .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                                .rounding(egui::Rounding::same(8.0))
                                .inner_margin(egui::Margin::same(12.0))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(
                                        egui::RichText::new("OLLAMA API ENDPOINT")
                                            .strong()
                                            .size(11.0)
                                            .color(tokens.text_secondary),
                                    );
                                    ui.add_space(8.0);

                                    egui::Grid::new("settings_connection_grid")
                                        .num_columns(2)
                                        .spacing([16.0, 10.0])
                                        .show(ui, |ui| {
                                            ui.label("Base URL:");
                                            ui.add(
                                                egui::TextEdit::singleline(&mut self.settings.ollama_url)
                                                    .desired_width(180.0)
                                                    .hint_text("http://127.0.0.1:11434"),
                                            );
                                            ui.end_row();
                                        });
                                });
                        }
                    }

                    ui.add_space(14.0);

                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add_sized(
                                    [90.0, 30.0],
                                    egui::Button::new(
                                        egui::RichText::new("Done")
                                            .strong()
                                            .color(egui::Color32::WHITE),
                                    )
                                    .fill(tokens.accent_primary)
                                    .rounding(egui::Rounding::same(8.0)),
                                )
                                .clicked()
                            {
                                self.show_settings = false;
                            }
                        });
                    });
                });
        }
    }
}

fn render_claude_message(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    msg: &Message,
    is_user: bool,
    is_last_assistant: bool,
    is_generating: bool,
    msg_idx: usize,
    tokens: &ThemeTokens,
) -> MessageAction {
    let mut action = MessageAction::None;

    if is_user {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                egui::Frame::none()
                    .fill(tokens.user_bubble_bg)
                    .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                    .rounding(egui::Rounding::same(14.0))
                    .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width() * 0.85);
                        ui.vertical(|ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(&msg.content).color(tokens.text_primary),
                                )
                                .wrap(),
                            );
                            ui.add_space(2.0);
                            ui.horizontal(|ui| {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("✏ Edit").clicked() {
                                            action =
                                                MessageAction::StartEdit(msg_idx, msg.content.clone());
                                        }
                                        if ui.small_button("📋").clicked() {
                                            ui.ctx().copy_text(msg.content.clone());
                                        }
                                    },
                                );
                            });
                        });
                    });
            });
        });
    } else {
        egui::Frame::none()
            .fill(tokens.assistant_bubble_bg)
            .inner_margin(egui::Margin::symmetric(4.0, 4.0))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("✦ Assistant")
                                .strong()
                                .size(12.0)
                                .color(tokens.accent_primary),
                        );
                    });

                    ui.add_space(4.0);

                    // Defer Markdown parsing during active generation for higher streaming frame rates
                    if is_generating && is_last_assistant {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&msg.content).color(tokens.text_primary),
                            )
                            .wrap(),
                        );
                    } else {
                        CommonMarkViewer::new().show(ui, cache, &msg.content);
                    }

                    ui.add_space(6.0);

                    if let (Some(comp), Some(secs)) = (msg.completion_tokens, msg.eval_duration_secs) {
                        let tps = if secs > 0.0 {
                            comp as f64 / secs
                        } else {
                            0.0
                        };
                        let prompt_str = msg
                            .prompt_tokens
                            .map(|p| format!(" (Prompt: {} | Response: {})", p, comp))
                            .unwrap_or_default();

                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "⚡ {:.1} t/s  ·  {:.2}s  ·  {} tokens{}",
                                    tps, secs, comp, prompt_str
                                ))
                                .size(11.0)
                                .color(tokens.text_secondary),
                            );

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("📋 Copy").clicked() {
                                    ui.ctx().copy_text(msg.content.clone());
                                }
                                if is_last_assistant && ui.small_button("🔄 Regenerate").clicked() {
                                    action = MessageAction::Regenerate(msg_idx);
                                }
                            });
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("📋 Copy").clicked() {
                                    ui.ctx().copy_text(msg.content.clone());
                                }
                                if is_last_assistant && ui.small_button("🔄 Regenerate").clicked() {
                                    action = MessageAction::Regenerate(msg_idx);
                                }
                            });
                        });
                    }
                });
            });
    }

    action
}

// ---------------------------------------------------------------------------
// Persistence & HTTP Network Workers
// ---------------------------------------------------------------------------

fn get_chats_path() -> PathBuf {
    if let Some(proj_dirs) = ProjectDirs::from("com", "OllamaStudio", "OllamaStudio") {
        let config_dir = proj_dirs.config_dir();
        let _ = std::fs::create_dir_all(config_dir);
        return config_dir.join("chats.json");
    }
    PathBuf::from("chats.json")
}

fn save_chats_sync(tabs: &[ChatTab]) {
    if let Ok(data) = serde_json::to_string_pretty(tabs) {
        let path = get_chats_path();
        let _ = std::fs::write(path, data);
    }
}

fn save_chats_async(tabs: Vec<ChatTab>) {
    thread::spawn(move || {
        save_chats_sync(&tabs);
    });
}

fn load_chats() -> Vec<ChatTab> {
    let path = get_chats_path();
    if let Ok(data) = std::fs::read_to_string(path) {
        if let Ok(tabs) = serde_json::from_str::<Vec<ChatTab>>(&data) {
            if !tabs.is_empty() {
                return tabs;
            }
        }
    }
    vec![ChatTab {
        id: 0,
        title: "New Chat".to_string(),
        messages: Vec::new(),
        selected_model: "llama3.2-vision:latest".to_string(),
        system_prompt: String::new(),
        temperature: default_temp(),
        top_p: default_top_p(),
    }]
}

#[derive(Serialize)]
struct OllamaUnloadRequest {
    model: String,
    keep_alive: i32,
}

fn unload_ollama_model(client: &reqwest::blocking::Client, base_url: &str, model: &str) {
    let payload = OllamaUnloadRequest {
        model: model.to_string(),
        keep_alive: 0,
    };
    let url = format!("{}/api/generate", base_url);
    let _ = client.post(url).json(&payload).send();
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<ModelInfo>,
}

fn fetch_ollama_models(client: &reqwest::blocking::Client, base_url: &str) -> Vec<String> {
    let url = format!("{}/api/tags", base_url);
    if let Ok(res) = client.get(url).send() {
        if let Ok(json) = res.json::<OllamaTagsResponse>() {
            let names: Vec<String> = json.models.into_iter().map(|m| m.name).collect();
            if !names.is_empty() {
                return names;
            }
        }
    }
    vec!["llama3.2-vision:latest".to_string(), "llama3.2:1b".to_string()]
}

#[derive(Deserialize)]
struct OllamaPsResponse {
    models: Vec<RunningModel>,
}

fn fetch_ollama_ps(client: &reqwest::blocking::Client, base_url: &str) -> Vec<RunningModel> {
    let url = format!("{}/api/ps", base_url);
    if let Ok(res) = client.get(url).send() {
        if let Ok(json) = res.json::<OllamaPsResponse>() {
            return json.models;
        }
    }
    Vec::new()
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    top_p: f32,
}

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct OllamaChatChunkResponse {
    message: Option<OllamaChatMessage>,
    done: Option<bool>,
    prompt_eval_count: Option<usize>,
    eval_count: Option<usize>,
    eval_duration: Option<u64>,
}

fn query_ollama_stream(
    client: Arc<reqwest::blocking::Client>,
    base_url: &str,
    model: &str,
    system_prompt: String,
    temperature: f32,
    top_p: f32,
    mut history: Vec<Message>,
    file_path: Option<String>,
    tab_id: usize,
    tx: &Sender<AppEvent>,
    ctx: &egui::Context,
    cancel_flag: Arc<AtomicBool>,
) {
    if history.len() > MAX_HISTORY_MESSAGES {
        let remove_count = history.len() - MAX_HISTORY_MESSAGES;
        history.drain(0..remove_count);
    }

    let mut chat_messages: Vec<OllamaChatMessage> = Vec::new();

    if !system_prompt.trim().is_empty() {
        chat_messages.push(OllamaChatMessage {
            role: "system".to_string(),
            content: system_prompt,
            images: None,
        });
    }

    for m in history {
        chat_messages.push(OllamaChatMessage {
            role: m.role,
            content: m.content,
            images: None,
        });
    }

    let mut image_payload: Option<Vec<String>> = None;
    if let Some(ref path_str) = file_path {
        let path = std::path::Path::new(path_str);
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif", "svg"].contains(&ext.as_str()) {
            match std::fs::read(path_str) {
                Ok(bytes) => {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
                    image_payload = Some(vec![b64]);
                }
                Err(err) => {
                    let _ = tx.send(AppEvent::StreamError {
                        tab_id,
                        error: format!("Failed to read image file '{}': {}", path_str, err),
                    });
                    ctx.request_repaint();
                    return;
                }
            }
        } else {
            match std::fs::read_to_string(path_str) {
                Ok(content) => {
                    if let Some(last_user) = chat_messages.iter_mut().rev().find(|m| m.role == "user") {
                        last_user.content = format!(
                            "Contents of {}:\n{}\n\nUser Prompt: {}",
                            path_str, content, last_user.content
                        );
                    }
                }
                Err(err) => {
                    let _ = tx.send(AppEvent::StreamError {
                        tab_id,
                        error: format!("Failed to read attachment '{}': {}", path_str, err),
                    });
                    ctx.request_repaint();
                    return;
                }
            }
        }
    }

    if let Some(last_user) = chat_messages.iter_mut().rev().find(|m| m.role == "user") {
        if image_payload.is_some() {
            last_user.images = image_payload;
        }
    }

    let payload = OllamaChatRequest {
        model: model.to_string(),
        messages: chat_messages,
        stream: true,
        options: OllamaOptions {
            temperature,
            top_p,
        },
    };

    let url = format!("{}/api/chat", base_url);
    let start_time = Instant::now();
    let mut last_repaint = Instant::now();
    let res = client.post(url).json(&payload).send();

    match res {
        Ok(response) => {
            let mut reader = BufReader::new(response);
            let mut line_buffer = String::new();

            let mut final_prompt_tokens = None;
            let mut final_completion_tokens = None;
            let mut final_eval_duration_secs = None;

            loop {
                if cancel_flag.load(Ordering::Relaxed) {
                    drop(reader);
                    unload_ollama_model(&client, base_url, model);
                    let _ = tx.send(AppEvent::StreamFinished {
                        tab_id,
                        prompt_tokens: None,
                        completion_tokens: None,
                        eval_duration_secs: None,
                    });
                    ctx.request_repaint();
                    return;
                }

                line_buffer.clear();
                match reader.read_line(&mut line_buffer) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(chunk) =
                            serde_json::from_str::<OllamaChatChunkResponse>(&line_buffer)
                        {
                            let chunk_content = chunk
                                .message
                                .as_ref()
                                .map(|m| m.content.clone())
                                .unwrap_or_default();

                            let _ = tx.send(AppEvent::StreamChunk {
                                tab_id,
                                chunk: chunk_content,
                            });

                            if last_repaint.elapsed() >= Duration::from_millis(33) {
                                ctx.request_repaint();
                                last_repaint = Instant::now();
                            }

                            if chunk.done.unwrap_or(false) {
                                final_prompt_tokens = chunk.prompt_eval_count;
                                final_completion_tokens = chunk.eval_count;
                                final_eval_duration_secs = chunk
                                    .eval_duration
                                    .map(|ns| ns as f64 / 1e9)
                                    .or_else(|| Some(start_time.elapsed().as_secs_f64()));
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx.send(AppEvent::StreamError {
                            tab_id,
                            error: format!("Read error: {}", err),
                        });
                        ctx.request_repaint();
                        return;
                    }
                }
            }
            let _ = tx.send(AppEvent::StreamFinished {
                tab_id,
                prompt_tokens: final_prompt_tokens,
                completion_tokens: final_completion_tokens,
                eval_duration_secs: final_eval_duration_secs,
            });
            ctx.request_repaint();
        }
        Err(_) => {
            let _ = tx.send(AppEvent::StreamError {
                tab_id,
                error: format!("Connection failed. Ensure Ollama is active at {}", base_url),
            });
            ctx.request_repaint();
        }
    }
}

// ---------------------------------------------------------------------------
// Entry Point
// ---------------------------------------------------------------------------

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1080.0, 760.0])
            .with_min_inner_size([680.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Ollama Studio",
        options,
        Box::new(|cc| Ok(Box::new(OllamaApp::new(cc)))),
    )
}
