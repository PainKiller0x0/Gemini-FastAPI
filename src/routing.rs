#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationRoute {
    ImageTool,
    VisionWorker,
    GeminiWeb,
}

pub struct RouteRequest<'a> {
    pub image_generation_enabled: bool,
    pub vision_worker_enabled: bool,
    pub has_attachments: bool,
    pub model: &'a str,
    pub latest_user_text: &'a str,
}

pub fn choose_generation_route(request: RouteRequest<'_>) -> GenerationRoute {
    if should_use_image_tool(
        request.image_generation_enabled,
        request.model,
        request.latest_user_text,
    ) {
        return GenerationRoute::ImageTool;
    }
    if request.has_attachments && request.vision_worker_enabled {
        return GenerationRoute::VisionWorker;
    }
    GenerationRoute::GeminiWeb
}

pub fn image_tool_prompt(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if let Some(start) = trimmed.rfind("<|im_start|>user") {
        let after_start = &trimmed[start + "<|im_start|>user".len()..];
        let after_start = after_start.strip_prefix('\n').unwrap_or(after_start);
        let end = after_start.find("<|im_end|>").unwrap_or(after_start.len());
        let candidate = after_start[..end].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    trimmed.to_string()
}

fn should_use_image_tool(image_generation_enabled: bool, model: &str, prompt: &str) -> bool {
    if !image_generation_enabled {
        return false;
    }
    let model = model.to_ascii_lowercase();
    if model.contains("image") || model.contains("imagen") {
        return true;
    }
    let latest_user_prompt = image_tool_prompt(prompt);
    explicit_image_generation_prompt(&latest_user_prompt)
}

pub fn explicit_image_generation_prompt(prompt: &str) -> bool {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return false;
    }
    let prompt_lc = prompt.to_ascii_lowercase();
    let negative = [
        "do not draw",
        "don't draw",
        "dont draw",
        "no image",
        "\u{4e0d}\u{8981}\u{753b}",
        "\u{4e0d}\u{7528}\u{753b}",
        "\u{522b}\u{753b}",
        "\u{4e0d}\u{8981}\u{751f}\u{6210}",
        "\u{4e0d}\u{7528}\u{751f}\u{6210}",
        "\u{522b}\u{751f}\u{6210}",
    ];
    if negative
        .iter()
        .any(|needle| prompt_lc.contains(needle) || prompt.contains(needle))
    {
        return false;
    }

    let english_intent = [
        "generate an image",
        "generate image",
        "create an image",
        "create image",
        "make an image",
        "draw an image",
        "draw a picture",
        "generate a picture",
        "create a picture",
        "draw me a",
    ];
    if english_intent
        .iter()
        .any(|needle| prompt_lc.contains(needle))
    {
        return true;
    }

    let chinese_intent = [
        "\u{7ed9}\u{6211}\u{753b}\u{4e00}\u{5f20}",
        "\u{5e2e}\u{6211}\u{753b}\u{4e00}\u{5f20}",
        "\u{5e2e}\u{6211}\u{753b}\u{5f20}",
        "\u{5e2e}\u{6211}\u{753b}\u{4e2a}",
        "\u{8bf7}\u{7ed9}\u{6211}\u{753b}",
        "\u{8bf7}\u{5e2e}\u{6211}\u{753b}",
        "\u{753b}\u{4e00}\u{5f20}",
        "\u{753b}\u{5f20}",
        "\u{753b}\u{4e2a}",
        "\u{751f}\u{6210}\u{4e00}\u{5f20}\u{56fe}",
        "\u{751f}\u{6210}\u{4e00}\u{5f20}\u{56fe}\u{7247}",
        "\u{5e2e}\u{6211}\u{751f}\u{6210}\u{4e00}\u{5f20}",
        "\u{7ed9}\u{6211}\u{751f}\u{6210}\u{4e00}\u{5f20}",
        "\u{751f}\u{6210}\u{56fe}\u{7247}",
        "\u{751f}\u{6210}\u{56fe}\u{50cf}",
        "\u{5e2e}\u{6211}\u{751f}\u{56fe}",
        "\u{7ed9}\u{6211}\u{751f}\u{56fe}",
        "\u{8bf7}\u{751f}\u{56fe}",
        "\u{751f}\u{4e00}\u{5f20}\u{56fe}",
        "\u{505a}\u{4e00}\u{5f20}\u{56fe}",
        "\u{5236}\u{4f5c}\u{4e00}\u{5f20}\u{56fe}",
        "\u{7ed8}\u{5236}\u{4e00}\u{5f20}",
    ];
    chinese_intent.iter().any(|needle| prompt.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::{
        GenerationRoute, RouteRequest, choose_generation_route, explicit_image_generation_prompt,
        image_tool_prompt,
    };

    #[test]
    fn chooses_image_tool_only_for_latest_explicit_image_request() {
        let prompt = "<|im_start|>user\n\u{7ed9}\u{6211}\u{753b}\u{4e00}\u{5f20}\u{7ea2}\u{8272}\u{5706}\u{5f62}\u{56fe}\u{6807}\n<|im_end|>\n<|im_start|>assistant\nok\n<|im_end|>\n<|im_start|>user\n\u{751f}\u{56fe}\u{8fd8}\u{6709}\u{62a5}\u{9519}\u{ff0c}\u{770b}\u{770b}\u{600e}\u{4e48}\u{529e}\n<|im_end|>\n<|im_start|>assistant";
        let latest = image_tool_prompt(prompt);
        assert_eq!(
            latest,
            "\u{751f}\u{56fe}\u{8fd8}\u{6709}\u{62a5}\u{9519}\u{ff0c}\u{770b}\u{770b}\u{600e}\u{4e48}\u{529e}"
        );
        assert!(!explicit_image_generation_prompt(&latest));
    }

    #[test]
    fn chooses_vision_worker_for_attachments_when_not_image_generation() {
        assert_eq!(
            choose_generation_route(RouteRequest {
                image_generation_enabled: true,
                vision_worker_enabled: true,
                has_attachments: true,
                model: "gemini-3.5-flash",
                latest_user_text: "\u{56fe}\u{91cc}\u{5199}\u{4e86}\u{4ec0}\u{4e48}\u{ff1f}",
            }),
            GenerationRoute::VisionWorker
        );
    }

    #[test]
    fn explicit_image_generation_prompt_is_strict() {
        assert!(explicit_image_generation_prompt(
            "\u{7ed9}\u{6211}\u{753b}\u{4e00}\u{5f20}\u{7ea2}\u{8272}\u{5706}\u{5f62}\u{56fe}\u{6807}"
        ));
        assert!(explicit_image_generation_prompt(
            "create an image of a red circle"
        ));
        assert!(!explicit_image_generation_prompt(
            "\u{56fe}\u{91cc}\u{5199}\u{4e86}\u{4ec0}\u{4e48}\u{ff1f}"
        ));
        assert!(!explicit_image_generation_prompt(
            "\u{5e2e}\u{6211}\u{770b}\u{770b}\u{8fd9}\u{5f20}\u{56fe}\u{7247}"
        ));
        assert!(!explicit_image_generation_prompt(
            "\u{8fd9}\u{4e2a}\u{9875}\u{9762}\u{753b}\u{9762}\u{5f88}\u{602a}"
        ));
        assert!(!explicit_image_generation_prompt(
            "\u{4e0d}\u{8981}\u{753b}\u{56fe}\u{ff0c}\u{7ed9}\u{6211}\u{6587}\u{5b57}\u{65b9}\u{6848}"
        ));
        assert!(!explicit_image_generation_prompt(
            "\u{751f}\u{56fe}\u{8fd8}\u{6709}\u{62a5}\u{9519}\u{ff0c}\u{770b}\u{770b}\u{600e}\u{4e48}\u{529e}"
        ));
    }
}
