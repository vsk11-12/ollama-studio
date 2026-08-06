use base64::Engine;
use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

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
            bg_base: egui::Color32::from_rgb(18, 18, 22),
            bg_surface: egui::Color32::from_rgb(24, 24, 28),
            bg_subtle: egui::Color32::from_rgb(32, 32, 38),
            border_subtle: egui::Color32::from_rgb(42, 42, 50),
            border_strong: egui::Color32::from_rgb(63, 63, 74),
            accent_primary: egui::Color32::from_rgb(99, 102, 241),
            text_primary: egui::Color32::from_rgb(244, 244, 245),
            text_secondary: egui::Color32::from_rgb(161, 161, 170),
            user_bubble_bg: egui::Color32::from_rgb(79, 70, 229),
            assistant_bubble_bg: egui::Color32::from_rgb(28, 28, 34),
        }
    }

    fn light() -> Self {
        Self {
            bg_base: egui::Color32::from_rgb(250, 250, 250),
            bg_surface: egui::Color32::from_rgb(255, 255, 255),
            bg_subtle: egui::Color32::from_rgb(244, 244, 245),
            border_subtle: egui::Color32::from_rgb(228, 228, 231),
            border_strong: egui::Color32::from_rgb(212, 212, 216),
            accent_primary: egui::Color32::from_rgb(79, 70, 229),
            text_primary: egui::Color32::from_rgb(24, 24, 27),
            text_secondary: egui::Color32::from_rgb(113, 113, 122),
            user_bubble_bg: egui::Color32::from_rgb(79, 70, 229),
            assistant_bubble_bg: egui::Color32::from_rgb(244, 244, 245),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
struct RunningModel {
    name: String,
    size: u64,
    size_vram: u64,
}

enum AppEvent {
    ModelsFetched(Vec<String>),
    RunningModelsFetched(Vec<RunningModel>),
    StreamChunk {
        tab_id: usize,
        chunk: String,
        prompt_tokens: Option<usize>,
        completion_tokens: Option<usize>,
    },
    StreamFinished {
        #[allow(dead_code)]
        tab_id: usize,
    },
    StreamError {
        tab_id: usize,
        error: String,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct ChatTab {
    id: usize,
    title: String,
    messages: Vec<Message>,
    selected_model: String,
}

enum MessageAction {
    None,
    StartEdit(usize, String),
    SaveEdit(usize, String),
    CancelEdit,
}

struct OllamaApp {
    tabs: Vec<ChatTab>,
    active_tab_id: usize,
    next_tab_id: usize,
    available_models: Vec<String>,
    running_models: Vec<RunningModel>,
    input_text: String,
    selected_file: Option<String>,
    scroll_to_bottom: bool,
    sidebar_collapsed: bool,
    editing_tab_id: Option<usize>,
    rename_input: String,
    active_view: ActiveView,

    // Token Counters
    total_prompt_tokens: usize,
    total_completion_tokens: usize,

    // Settings Modal State
    show_settings: bool,
    ollama_url: String,
    dark_mode: bool,
    zoom_factor: f32,
    selected_font: SelectedFont,
    base_font_size: f32,

    markdown_cache: CommonMarkCache,
    editing_msg: Option<(usize, usize, String)>,
    cancel_flag: Option<Arc<AtomicBool>>,
    is_generating: bool,
    tx: Sender<AppEvent>,
    rx: Receiver<AppEvent>,
}

impl OllamaApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();

        let tx_clone = tx.clone();
        let ctx = cc.egui_ctx.clone();
        let ollama_url = DEFAULT_OLLAMA_URL.to_string();
        let url_clone = ollama_url.clone();

        thread::spawn(move || {
            let models = fetch_ollama_models(&url_clone);
            let _ = tx_clone.send(AppEvent::ModelsFetched(models));
            ctx.request_repaint();
        });

        let initial_tabs = load_chats();
        let active_id = initial_tabs.first().map(|t| t.id).unwrap_or(0);
        let max_id = initial_tabs.iter().map(|t| t.id).max().unwrap_or(0);

        Self {
            tabs: initial_tabs,
            active_tab_id: active_id,
            next_tab_id: max_id + 1,
            available_models: vec!["llama3.2-vision:latest".to_string(), "llama3.2:1b".to_string()],
            running_models: Vec::new(),
            input_text: String::new(),
            selected_file: None,
            scroll_to_bottom: false,
            sidebar_collapsed: false,
            editing_tab_id: None,
            rename_input: String::new(),
            active_view: ActiveView::Chat,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            show_settings: false,
            ollama_url,
            dark_mode: true,
            zoom_factor: 1.0,
            selected_font: SelectedFont::Proportional,
            base_font_size: 14.0,
            markdown_cache: CommonMarkCache::default(),
            editing_msg: None,
            cancel_flag: None,
            is_generating: false,
            tx,
            rx,
        }
    }

    fn stop_generation(&mut self) {
        if let Some(flag) = &self.cancel_flag {
            flag.store(true, Ordering::Relaxed);
        }

        if let Some(idx) = self.get_active_tab_index() {
            let model = self.tabs[idx].selected_model.clone();
            let url = self.ollama_url.clone();
            thread::spawn(move || {
                unload_ollama_model(&url, &model);
            });
        }

        self.is_generating = false;
    }

    fn start_generation(
        &mut self,
        ctx: &egui::Context,
        active_idx: usize,
        images: Option<Vec<String>>,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag = Some(cancel.clone());
        self.is_generating = true;
        self.scroll_to_bottom = true;

        let model_name = self.tabs[active_idx].selected_model.clone();
        let tab_id = self.tabs[active_idx].id;
        let url = self.ollama_url.clone();

        let mut history = self.tabs[active_idx].messages.clone();
        if let Some(last) = history.last() {
            if last.role == "assistant" && last.content.is_empty() {
                history.pop();
            }
        }

        save_chats_async(self.tabs.clone());

        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();

        thread::spawn(move || {
            query_ollama_stream(&url, &model_name, history, images, tab_id, &tx, &ctx_clone, cancel);
        });
    }

    fn fetch_running_models(&mut self, ctx: &egui::Context) {
        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();
        let url = self.ollama_url.clone();
        thread::spawn(move || {
            let running = fetch_ollama_ps(&url);
            let _ = tx.send(AppEvent::RunningModelsFetched(running));
            ctx_clone.request_repaint();
        });
    }

    fn unload_all_models(&mut self, ctx: &egui::Context) {
        let url = self.ollama_url.clone();
        let models = self.running_models.clone();
        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();

        thread::spawn(move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default();

            for m in models {
                let _ = client
                    .post(format!("{}/api/generate", url))
                    .json(&serde_json::json!({
                        "model": m.name,
                        "keep_alive": 0
                    }))
                    .send();
            }

            let running = fetch_ollama_ps(&url);
            let _ = tx.send(AppEvent::RunningModelsFetched(running));
            ctx_clone.request_repaint();
        });
    }

    fn apply_theme_and_scale(&mut self, ctx: &egui::Context) {
        ctx.set_zoom_factor(self.zoom_factor);

        let font_family = match self.selected_font {
            SelectedFont::Proportional => egui::FontFamily::Proportional,
            SelectedFont::Monospace => egui::FontFamily::Monospace,
        };

        let mut style = (*ctx.style()).clone();
        style.text_styles = [
            (
                egui::TextStyle::Heading,
                egui::FontId::new(self.base_font_size + 4.0, font_family.clone()),
            ),
            (
                egui::TextStyle::Body,
                egui::FontId::new(self.base_font_size, font_family.clone()),
            ),
            (
                egui::TextStyle::Button,
                egui::FontId::new(self.base_font_size - 1.0, font_family.clone()),
            ),
            (
                egui::TextStyle::Small,
                egui::FontId::new(self.base_font_size - 3.0, font_family.clone()),
            ),
        ]
        .into();
        ctx.set_style(style);

        let tokens = if self.dark_mode {
            ThemeTokens::dark()
        } else {
            ThemeTokens::light()
        };

        let mut visuals = if self.dark_mode {
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
        visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);

        visuals.widgets.hovered.bg_fill = if self.dark_mode {
            egui::Color32::from_rgb(45, 45, 56)
        } else {
            egui::Color32::from_rgb(238, 238, 242)
        };
        visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, tokens.text_primary);
        visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);

        visuals.widgets.active.bg_fill = tokens.accent_primary;
        visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
        visuals.widgets.active.rounding = egui::Rounding::same(6.0);

        visuals.selection.bg_fill = tokens.accent_primary;
        visuals.window_stroke = egui::Stroke::new(1.0, tokens.border_subtle);
        visuals.window_rounding = egui::Rounding::same(10.0);

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
                    self.running_models = models;
                }
                AppEvent::StreamChunk {
                    tab_id,
                    chunk,
                    prompt_tokens,
                    completion_tokens,
                } => {
                    if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) {
                        if let Some(last_msg) = tab.messages.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.content.push_str(&chunk);
                            }
                        }
                    }
                    if let Some(p) = prompt_tokens {
                        self.total_prompt_tokens += p;
                    }
                    if let Some(c) = completion_tokens {
                        self.total_completion_tokens += c;
                    }
                    self.scroll_to_bottom = true;
                }
                AppEvent::StreamFinished { tab_id: _ } => {
                    self.is_generating = false;
                    self.scroll_to_bottom = true;
                    save_chats_async(self.tabs.clone());
                }
                AppEvent::StreamError { tab_id, error } => {
                    if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) {
                        if let Some(last_msg) = tab.messages.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.content = format!("Error: {}", error);
                            }
                        }
                    }
                    self.is_generating = false;
                    self.scroll_to_bottom = true;
                    save_chats_async(self.tabs.clone());
                }
            }
        }
    }

    fn get_active_tab_index(&self) -> Option<usize> {
        self.tabs.iter().position(|t| t.id == self.active_tab_id)
    }

    fn render_stats_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let tokens = if self.dark_mode {
            ThemeTokens::dark()
        } else {
            ThemeTokens::light()
        };

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.add_space(4.0);

                // --- Header & Actions Toolbar ---
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.heading(
                            egui::RichText::new("System & Performance Statistics")
                                .strong()
                                .size(18.0)
                                .color(tokens.text_primary),
                        );
                        ui.label(
                            egui::RichText::new(
                                "Monitor session throughput and local LLM memory allocations.",
                            )
                            .size(12.0)
                            .color(tokens.text_secondary),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_sized(
                                [120.0, 30.0],
                                egui::Button::new(
                                    egui::RichText::new("🔄 Refresh Stats")
                                        .size(12.0)
                                        .color(tokens.text_primary),
                                )
                                .fill(tokens.bg_subtle)
                                .rounding(egui::Rounding::same(6.0)),
                            )
                            .clicked()
                        {
                            self.fetch_running_models(ctx);
                        }

                        if ui
                            .add_sized(
                                [160.0, 30.0],
                                egui::Button::new(
                                    egui::RichText::new("⏹ Stop & Unload Models")
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(239, 68, 68)),
                                )
                                .fill(tokens.bg_subtle)
                                .rounding(egui::Rounding::same(6.0)),
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
                ui.separator();
                ui.add_space(16.0);

                // --- Metric Summary Cards ---
                ui.horizontal_top(|ui| {
                    let card_width = ((ui.available_width() - 24.0) / 3.0).max(180.0);

                    // Prompt Tokens Card
                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(8.0))
                        .inner_margin(egui::Margin::same(14.0))
                        .show(ui, |ui| {
                            ui.set_width(card_width);
                            ui.label(
                                egui::RichText::new("PROMPT TOKENS")
                                    .size(11.0)
                                    .strong()
                                    .color(tokens.text_secondary),
                            );
                            ui.add_space(6.0);
                            ui.heading(
                                egui::RichText::new(format!("{}", self.total_prompt_tokens))
                                    .size(24.0)
                                    .strong()
                                    .color(tokens.accent_primary),
                            );
                        });

                    ui.add_space(12.0);

                    // Completion Tokens Card
                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(8.0))
                        .inner_margin(egui::Margin::same(14.0))
                        .show(ui, |ui| {
                            ui.set_width(card_width);
                            ui.label(
                                egui::RichText::new("COMPLETION TOKENS")
                                    .size(11.0)
                                    .strong()
                                    .color(tokens.text_secondary),
                            );
                            ui.add_space(6.0);
                            ui.heading(
                                egui::RichText::new(format!("{}", self.total_completion_tokens))
                                    .size(24.0)
                                    .strong()
                                    .color(tokens.accent_primary),
                            );
                        });

                    ui.add_space(12.0);

                    // Total Session Tokens Card
                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(8.0))
                        .inner_margin(egui::Margin::same(14.0))
                        .show(ui, |ui| {
                            ui.set_width(card_width);
                            ui.label(
                                egui::RichText::new("TOTAL SESSION TOKENS")
                                    .size(11.0)
                                    .strong()
                                    .color(tokens.text_secondary),
                            );
                            ui.add_space(6.0);
                            ui.heading(
                                egui::RichText::new(format!(
                                    "{}",
                                    self.total_prompt_tokens + self.total_completion_tokens
                                ))
                                .size(24.0)
                                .strong()
                                .color(tokens.accent_primary),
                            );
                        });
                });

                ui.add_space(20.0);

                // --- Active / Loaded Models Card ---
                egui::Frame::none()
                    .fill(tokens.bg_surface)
                    .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                    .rounding(egui::Rounding::same(10.0))
                    .inner_margin(egui::Margin::same(16.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.heading(
                                egui::RichText::new("🧠 Loaded Models in RAM / VRAM")
                                    .size(14.0)
                                    .strong()
                                    .color(tokens.text_primary),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{} Active", self.running_models.len()))
                                        .size(12.0)
                                        .color(tokens.text_secondary),
                                );
                            });
                        });
                        ui.add_space(12.0);

                        if self.running_models.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(8.0);
                                ui.label(
                                    egui::RichText::new("No LLMs currently residing in active memory.")
                                        .italics()
                                        .size(13.0)
                                        .color(tokens.text_secondary),
                                );
                                ui.add_space(8.0);
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
                                        egui::RichText::new("VRAM Allocation")
                                            .strong()
                                            .color(tokens.text_secondary),
                                    );
                                    ui.end_row();

                                    for model in &self.running_models {
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
                                                .text(format!("{:.0}% VRAM", ratio * 100.0))
                                                .desired_width(130.0),
                                        );
                                        ui.end_row();
                                    }
                                });
                        }
                    });

                ui.add_space(20.0);

                // --- Installed Models List Card ---
                egui::Frame::none()
                    .fill(tokens.bg_surface)
                    .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                    .rounding(egui::Rounding::same(10.0))
                    .inner_margin(egui::Margin::same(16.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.heading(
                            egui::RichText::new("📦 Installed Local Models")
                                .size(14.0)
                                .strong()
                                .color(tokens.text_primary),
                        );
                        ui.add_space(10.0);

                        if self.available_models.is_empty() {
                            ui.label(
                                egui::RichText::new("No models detected from Ollama.")
                                    .italics()
                                    .color(tokens.text_secondary),
                            );
                        } else {
                            for model in &self.available_models {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("⚡")
                                            .size(12.0)
                                            .color(tokens.accent_primary),
                                    );
                                    ui.label(
                                        egui::RichText::new(model)
                                            .size(13.0)
                                            .color(tokens.text_primary),
                                    );
                                });
                                ui.add_space(2.0);
                            }
                        }
                    });
            });
    }
}

impl eframe::App for OllamaApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        save_chats_sync(&self.tabs);
        if let Some(idx) = self.get_active_tab_index() {
            unload_ollama_model(&self.ollama_url, &self.tabs[idx].selected_model);
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_events();
        self.apply_theme_and_scale(ctx);

        let tokens = if self.dark_mode {
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
                            .rounding(egui::Rounding::same(6.0)),
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
                                    egui::RichText::new("Workspace")
                                        .strong()
                                        .size(14.0)
                                        .color(tokens.text_primary),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.button("<").on_hover_text("Hide Sidebar").clicked() {
                                            self.sidebar_collapsed = true;
                                        }
                                    },
                                );
                            });

                            ui.add_space(10.0);

                            // Mode Selector (Chats vs Stats)
                            ui.horizontal(|ui| {
                                let chat_active = self.active_view == ActiveView::Chat;
                                let stats_active = self.active_view == ActiveView::Stats;

                                if ui
                                    .selectable_label(chat_active, "💬 Chats")
                                    .clicked()
                                {
                                    self.active_view = ActiveView::Chat;
                                }
                                if ui
                                    .selectable_label(stats_active, "📊 Stats")
                                    .clicked()
                                {
                                    self.active_view = ActiveView::Stats;
                                    self.fetch_running_models(ctx);
                                }
                            });

                            ui.add_space(8.0);

                            if self.active_view == ActiveView::Chat {
                                let new_chat_btn = ui.add_sized(
                                    [ui.available_width(), 34.0],
                                    egui::Button::new(
                                        egui::RichText::new("+ New Chat")
                                            .strong()
                                            .size(13.0)
                                            .color(egui::Color32::WHITE),
                                    )
                                    .fill(tokens.accent_primary)
                                    .rounding(egui::Rounding::same(8.0)),
                                );

                                if new_chat_btn.clicked() {
                                    if self.is_generating {
                                        self.stop_generation();
                                    }
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
                                        messages: vec![Message {
                                            role: "assistant".to_string(),
                                            content: "Started a new session.".to_string(),
                                        }],
                                        selected_model: fallback_model,
                                    });
                                    self.active_tab_id = new_id;
                                    save_chats_async(self.tabs.clone());
                                }

                                ui.add_space(14.0);
                                ui.label(
                                    egui::RichText::new("RECENT CONVERSATIONS")
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
                                                if ui.button("Save").clicked() {
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
                                                                self.active_tab_id = tab.id;
                                                            }

                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    if tabs_len > 1
                                                                        && ui
                                                                            .small_button("x")
                                                                            .clicked()
                                                                    {
                                                                        tab_to_close =
                                                                            Some(tab.id);
                                                                    }
                                                                    if ui
                                                                        .small_button("Edit")
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
                                        save_chats_async(self.tabs.clone());
                                    }

                                    if let Some(id) = tab_to_close {
                                        self.tabs.retain(|t| t.id != id);
                                        if self.active_tab_id == id {
                                            if let Some(first) = self.tabs.first() {
                                                self.active_tab_id = first.id;
                                            }
                                        }
                                        save_chats_async(self.tabs.clone());
                                    }
                                });
                            }
                        });
                    });
                });
        }

        // --- Main Canvas ---
        egui::CentralPanel::default()
            .frame(
                egui::Frame::central_panel(ctx.style().as_ref())
                    .fill(tokens.bg_base)
                    .inner_margin(egui::Margin::symmetric(24.0, 14.0)),
            )
            .show(ctx, |ui| {
                if self.active_view == ActiveView::Stats {
                    self.render_stats_tab(ui, ctx);
                    return;
                }

                // Chat View
                ui.horizontal(|ui| {
                    if self.sidebar_collapsed {
                        if ui.button(">").on_hover_text("Open Sidebar").clicked() {
                            self.sidebar_collapsed = false;
                        }
                        ui.add_space(8.0);
                    }

                    ui.heading(
                        egui::RichText::new("Ollama Studio")
                            .strong()
                            .color(tokens.text_primary),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Refresh").clicked() {
                            let tx_clone = self.tx.clone();
                            let ctx_clone = ctx.clone();
                            let url = self.ollama_url.clone();
                            thread::spawn(move || {
                                let models = fetch_ollama_models(&url);
                                let _ = tx_clone.send(AppEvent::ModelsFetched(models));
                                ctx_clone.request_repaint();
                            });
                        }

                        ui.add_space(4.0);

                        if ui.button("Clear").clicked() {
                            if self.is_generating {
                                self.stop_generation();
                            }
                            if let Some(idx) = self.get_active_tab_index() {
                                self.tabs[idx].messages.clear();
                                save_chats_async(self.tabs.clone());
                            }
                        }

                        ui.add_space(6.0);

                        if let Some(idx) = self.get_active_tab_index() {
                            let selected_text = self.tabs[idx].selected_model.clone();
                            egui::ComboBox::from_id_source("model_selector")
                                .selected_text(
                                    egui::RichText::new(&selected_text)
                                        .color(tokens.text_primary),
                                )
                                .show_ui(ui, |ui: &mut egui::Ui| {
                                    for model in &self.available_models {
                                        ui.selectable_value(
                                            &mut self.tabs[idx].selected_model,
                                            model.clone(),
                                            model,
                                        );
                                    }
                                });
                        }
                    });
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                let active_idx = match self.get_active_tab_index() {
                    Some(idx) => idx,
                    None => return,
                };

                let reserved_bottom_space = 105.0;
                let available_height = (ui.available_height() - reserved_bottom_space).max(100.0);

                let mut message_action = MessageAction::None;

                egui::ScrollArea::vertical()
                    .max_height(available_height)
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
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
                                let (_, _, ref mut edit_text) = self.editing_msg.as_mut().unwrap();

                                ui.horizontal(|ui| {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::TOP),
                                        |ui| {
                                            egui::Frame::none()
                                                .fill(tokens.bg_subtle)
                                                .stroke(egui::Stroke::new(
                                                    1.0,
                                                    tokens.border_strong,
                                                ))
                                                .rounding(egui::Rounding::same(12.0))
                                                .inner_margin(egui::Margin::same(12.0))
                                                .show(ui, |ui| {
                                                    ui.set_max_width(ui.available_width() * 0.75);
                                                    ui.vertical(|ui| {
                                                        ui.label("Edit Query");
                                                        ui.add_space(4.0);
                                                        ui.add(
                                                            egui::TextEdit::multiline(edit_text)
                                                                .desired_width(f32::INFINITY),
                                                        );
                                                        ui.add_space(6.0);
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Save & Resend").clicked()
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
                                        },
                                    );
                                });
                            } else {
                                ui.horizontal(|ui| {
                                    if is_user {
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::TOP),
                                            |ui| {
                                                if render_message_bubble(
                                                    ui,
                                                    &mut self.markdown_cache,
                                                    msg,
                                                    true,
                                                    &tokens,
                                                ) {
                                                    message_action = MessageAction::StartEdit(
                                                        msg_idx,
                                                        msg.content.clone(),
                                                    );
                                                }
                                            },
                                        );
                                    } else {
                                        ui.with_layout(
                                            egui::Layout::left_to_right(egui::Align::TOP),
                                            |ui| {
                                                render_message_bubble(
                                                    ui,
                                                    &mut self.markdown_cache,
                                                    msg,
                                                    false,
                                                    &tokens,
                                                );
                                            },
                                        );
                                    }
                                });
                            }
                            ui.add_space(12.0);
                        }

                        if self.is_generating {
                            ui.horizontal(|ui| {
                                ui.add(egui::Spinner::new());
                                ui.label(
                                    egui::RichText::new("Generating response...")
                                        .italics()
                                        .color(tokens.text_secondary),
                                );
                                ui.add_space(8.0);
                                if ui.button("Stop").clicked() {
                                    self.stop_generation();
                                }
                            });
                            ui.add_space(8.0);
                        }

                        if self.scroll_to_bottom {
                            ui.scroll_to_cursor(Some(egui::Align::Max));
                            self.scroll_to_bottom = false;
                        }
                    });

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
                            });
                        }
                        self.start_generation(ctx, active_idx, None);
                    }
                    MessageAction::None => {}
                }

                ui.add_space(6.0);

                // --- Input Box ---
                let total_avail_width = ui.available_width();
                let prompt_width = total_avail_width.min(840.0);

                ui.horizontal(|ui| {
                    let side_margin = ((total_avail_width - prompt_width) / 2.0).max(0.0);
                    if side_margin > 0.0 {
                        ui.add_space(side_margin);
                    }

                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(16.0))
                        .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                        .show(ui, |ui| {
                            ui.set_width(prompt_width);
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(!self.is_generating, egui::Button::new("Attach"))
                                    .clicked()
                                {
                                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                                        self.selected_file = Some(path.display().to_string());
                                    }
                                }

                                if let Some(file) = &self.selected_file {
                                    let filename = std::path::Path::new(file)
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("file");

                                    if ui.button(format!("File: {} x", filename)).clicked() {
                                        self.selected_file = None;
                                    }
                                }

                                let send_button_width = 46.0;
                                let available_for_text =
                                    (ui.available_width() - send_button_width - 8.0).max(100.0);

                                let text_edit = ui.add_sized(
                                    [available_for_text, 42.0],
                                    egui::TextEdit::multiline(&mut self.input_text)
                                        .hint_text(if self.is_generating {
                                            "Generating..."
                                        } else {
                                            "Send a message (Enter to send, Shift+Enter for newline)..."
                                        })
                                        .interactive(!self.is_generating)
                                        .frame(false),
                                );

                                let enter_pressed = text_edit.has_focus()
                                    && ui.input(|i| {
                                        i.key_pressed(egui::Key::Enter) && !i.modifiers.shift
                                    });

                                if self.is_generating {
                                    if ui.button("Stop").clicked() {
                                        self.stop_generation();
                                    }
                                } else {
                                    let send_btn = ui.add_sized(
                                        [32.0, 26.0],
                                        egui::Button::new(
                                            egui::RichText::new("Send")
                                                .strong()
                                                .color(egui::Color32::WHITE),
                                        )
                                        .fill(tokens.accent_primary)
                                        .rounding(egui::Rounding::same(8.0)),
                                    );

                                    if (send_btn.clicked() || enter_pressed)
                                        && !self.input_text.trim().is_empty()
                                    {
                                        let user_message = self.input_text.trim().to_string();
                                        let mut full_prompt = user_message.clone();
                                        let mut image_payload: Option<Vec<String>> = None;

                                        if let Some(file_path) = &self.selected_file {
                                            let path = std::path::Path::new(file_path);
                                            let ext = path
                                                .extension()
                                                .and_then(|s| s.to_str())
                                                .unwrap_or("")
                                                .to_lowercase();

                                            if [
                                                "png", "jpg", "jpeg", "webp", "bmp", "tiff",
                                                "gif", "svg",
                                            ]
                                            .contains(&ext.as_str())
                                            {
                                                if let Ok(bytes) = std::fs::read(file_path) {
                                                    let b64 =
                                                        base64::engine::general_purpose::STANDARD
                                                            .encode(bytes);
                                                    image_payload = Some(vec![b64]);
                                                }
                                            } else {
                                                if let Ok(content) =
                                                    std::fs::read_to_string(file_path)
                                                {
                                                    full_prompt = format!(
                                                        "Contents of {}:\n{}\n\nUser Prompt: {}",
                                                        file_path, content, user_message
                                                    );
                                                }
                                            }
                                        }

                                        self.tabs[active_idx].messages.push(Message {
                                            role: "user".to_string(),
                                            content: full_prompt,
                                        });

                                        self.tabs[active_idx].messages.push(Message {
                                            role: "assistant".to_string(),
                                            content: String::new(),
                                        });

                                        self.input_text.clear();
                                        self.selected_file = None;
                                        self.start_generation(ctx, active_idx, image_payload);
                                    }
                                }
                            });
                        });
                });
            });

        // --- Settings Modal Window ---
        if self.show_settings {
            egui::Window::new("Settings & Preferences")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .frame(
                    egui::Frame::window(ctx.style().as_ref())
                        .fill(tokens.bg_surface)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(12.0))
                        .inner_margin(egui::Margin::same(20.0)),
                )
                .show(ctx, |ui| {
                    ui.set_width(380.0);

                    ui.horizontal(|ui| {
                        ui.heading(
                            egui::RichText::new("Preferences")
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
                    ui.add_space(12.0);

                    // Section 1: Appearance
                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(8.0))
                        .inner_margin(egui::Margin::same(12.0))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(
                                egui::RichText::new("APPEARANCE & THEME")
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
                                        ui.selectable_value(&mut self.dark_mode, true, "🌙 Dark");
                                        ui.selectable_value(&mut self.dark_mode, false, "☀️ Light");
                                    });
                                    ui.end_row();

                                    ui.label("UI Scale:");
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::Slider::new(&mut self.zoom_factor, 0.75..=1.50)
                                                .step_by(0.05)
                                                .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)),
                                        );
                                        if ui.small_button("Reset").clicked() {
                                            self.zoom_factor = 1.0;
                                        }
                                    });
                                    ui.end_row();
                                });
                        });

                    ui.add_space(10.0);

                    // Section 2: Typography
                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(8.0))
                        .inner_margin(egui::Margin::same(12.0))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(
                                egui::RichText::new("TYPOGRAPHY & FONTS")
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
                                    egui::ComboBox::from_id_source("font_family_select")
                                        .selected_text(match self.selected_font {
                                            SelectedFont::Proportional => "Proportional (Sans)",
                                            SelectedFont::Monospace => "Monospace (Code)",
                                        })
                                        .show_ui(ui, |ui: &mut egui::Ui| {
                                            ui.selectable_value(
                                                &mut self.selected_font,
                                                SelectedFont::Proportional,
                                                "Proportional (Sans)",
                                            );
                                            ui.selectable_value(
                                                &mut self.selected_font,
                                                SelectedFont::Monospace,
                                                "Monospace (Code)",
                                            );
                                        });
                                    ui.end_row();

                                    ui.label("Font Size:");
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::Slider::new(&mut self.base_font_size, 11.0..=20.0)
                                                .suffix(" px"),
                                        );
                                        if ui.small_button("Reset").clicked() {
                                            self.base_font_size = 14.0;
                                        }
                                    });
                                    ui.end_row();
                                });
                        });

                    ui.add_space(10.0);

                    // Section 3: Ollama Connection
                    egui::Frame::none()
                        .fill(tokens.bg_subtle)
                        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
                        .rounding(egui::Rounding::same(8.0))
                        .inner_margin(egui::Margin::same(12.0))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(
                                egui::RichText::new("OLLAMA CONNECTION")
                                    .strong()
                                    .size(11.0)
                                    .color(tokens.text_secondary),
                            );
                            ui.add_space(8.0);

                            egui::Grid::new("settings_connection_grid")
                                .num_columns(2)
                                .spacing([16.0, 10.0])
                                .show(ui, |ui| {
                                    ui.label("Base Endpoint:");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.ollama_url)
                                            .desired_width(180.0)
                                            .hint_text("http://127.0.0.1:11434"),
                                    );
                                    ui.end_row();
                                });
                        });

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
                                    .rounding(egui::Rounding::same(6.0)),
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

fn render_message_bubble(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    msg: &Message,
    is_user: bool,
    tokens: &ThemeTokens,
) -> bool {
    let mut edit_requested = false;
    let max_width = ui.available_width() * 0.75;

    let (bg_color, stroke, text_color) = if is_user {
        (
            tokens.user_bubble_bg,
            egui::Stroke::NONE,
            egui::Color32::WHITE,
        )
    } else {
        (
            tokens.assistant_bubble_bg,
            egui::Stroke::new(1.0, tokens.border_subtle),
            tokens.text_primary,
        )
    };

    let rounding = egui::Rounding {
        nw: 14.0,
        ne: 14.0,
        sw: if is_user { 14.0 } else { 2.0 },
        se: if is_user { 2.0 } else { 14.0 },
    };

    egui::Frame::none()
        .fill(bg_color)
        .stroke(stroke)
        .rounding(rounding)
        .inner_margin(egui::Margin::symmetric(14.0, 12.0))
        .show(ui, |ui| {
            ui.set_max_width(max_width);
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    let header = if is_user { "You" } else { "Assistant" };
                    ui.label(
                        egui::RichText::new(header)
                            .strong()
                            .size(11.0)
                            .color(if is_user {
                                egui::Color32::from_rgb(224, 231, 255)
                            } else {
                                tokens.text_secondary
                            }),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Copy").clicked() {
                            ui.ctx().copy_text(msg.content.clone());
                        }

                        if is_user && ui.small_button("Edit").clicked() {
                            edit_requested = true;
                        }
                    });
                });
                ui.add_space(4.0);

                if is_user {
                    ui.label(egui::RichText::new(&msg.content).color(text_color));
                } else {
                    CommonMarkViewer::new("msg_markdown").show(ui, cache, &msg.content);
                }
            });
        });

    edit_requested
}

// ---------------- Persistence & HTTP Helpers ----------------

fn save_chats_sync(tabs: &[ChatTab]) {
    if let Ok(data) = serde_json::to_string_pretty(tabs) {
        let _ = std::fs::write("chats.json", data);
    }
}

fn save_chats_async(tabs: Vec<ChatTab>) {
    thread::spawn(move || {
        save_chats_sync(&tabs);
    });
}

fn load_chats() -> Vec<ChatTab> {
    if let Ok(data) = std::fs::read_to_string("chats.json") {
        if let Ok(tabs) = serde_json::from_str::<Vec<ChatTab>>(&data) {
            if !tabs.is_empty() {
                return tabs;
            }
        }
    }
    vec![ChatTab {
        id: 0,
        title: "New Chat".to_string(),
        messages: vec![Message {
            role: "assistant".to_string(),
            content: "Hello! How can I assist you today?".to_string(),
        }],
        selected_model: "llama3.2-vision:latest".to_string(),
    }]
}

#[derive(Serialize)]
struct OllamaUnloadRequest {
    model: String,
    keep_alive: i32,
}

fn unload_ollama_model(base_url: &str, model: &str) {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();

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

fn fetch_ollama_models(base_url: &str) -> Vec<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();

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

fn fetch_ollama_ps(base_url: &str) -> Vec<RunningModel> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();

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
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaChatChunkResponse {
    message: Option<OllamaChatMessage>,
    done: Option<bool>,
    prompt_eval_count: Option<usize>,
    eval_count: Option<usize>,
}

fn query_ollama_stream(
    base_url: &str,
    model: &str,
    history: Vec<Message>,
    images: Option<Vec<String>>,
    tab_id: usize,
    tx: &Sender<AppEvent>,
    ctx: &egui::Context,
    cancel_flag: Arc<AtomicBool>,
) {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            let _ = tx.send(AppEvent::StreamError {
                tab_id,
                error: err.to_string(),
            });
            ctx.request_repaint();
            return;
        }
    };

    let mut chat_messages: Vec<OllamaChatMessage> = history
        .into_iter()
        .map(|m| OllamaChatMessage {
            role: m.role,
            content: m.content,
            images: None,
        })
        .collect();

    if let Some(last_user) = chat_messages.iter_mut().rev().find(|m| m.role == "user") {
        last_user.images = images;
    }

    let payload = OllamaChatRequest {
        model: model.to_string(),
        messages: chat_messages,
        stream: true,
    };

    let url = format!("{}/api/chat", base_url);
    let res = client.post(url).json(&payload).send();

    match res {
        Ok(response) => {
            let mut reader = BufReader::new(response);
            let mut line_buffer = String::new();

            loop {
                if cancel_flag.load(Ordering::Relaxed) {
                    drop(reader);
                    unload_ollama_model(base_url, model);
                    let _ = tx.send(AppEvent::StreamFinished { tab_id });
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
                                prompt_tokens: chunk.prompt_eval_count,
                                completion_tokens: chunk.eval_count,
                            });
                            ctx.request_repaint();

                            if chunk.done.unwrap_or(false) {
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
            let _ = tx.send(AppEvent::StreamFinished { tab_id });
            ctx.request_repaint();
        }
        Err(_) => {
            let _ = tx.send(AppEvent::StreamError {
                tab_id,
                error: format!("Connection failed. Ensure Ollama is running at {}", base_url),
            });
            ctx.request_repaint();
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: viewport_builder(),
        ..Default::default()
    };
    eframe::run_native(
        "Ollama Studio",
        options,
        Box::new(|cc| Ok(Box::new(OllamaApp::new(cc)))),
    )
}

fn viewport_builder() -> egui::ViewportBuilder {
    egui::ViewportBuilder::default()
        .with_inner_size([1000.0, 720.0])
        .with_min_inner_size([680.0, 500.0])
}
