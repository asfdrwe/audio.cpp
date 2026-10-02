#include "engine/models/irodori_tts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::models::irodori_tts {
namespace json = engine::io::json;
namespace {

double optional_ratio(const json::Value &root, const std::string &key, double fallback) {
    const auto *value = root.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_number();
}

IrodoriModelConfig parse_model_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("model_config");

    IrodoriModelConfig config;
    config.latent_dim = json::optional_i64(root, "latent_dim", config.latent_dim);
    config.latent_patch_size = json::optional_i64(root, "latent_patch_size", config.latent_patch_size);
    config.model_dim = json::optional_i64(root, "model_dim", config.model_dim);
    config.num_layers = json::optional_i64(root, "num_layers", config.num_layers);
    config.num_heads = json::optional_i64(root, "num_heads", config.num_heads);
    config.mlp_ratio = optional_ratio(root, "mlp_ratio", config.mlp_ratio);
    config.text_mlp_ratio = optional_ratio(root, "text_mlp_ratio", config.mlp_ratio);
    config.speaker_mlp_ratio = optional_ratio(root, "speaker_mlp_ratio", config.mlp_ratio);
    config.dropout = json::optional_f32(root, "dropout", config.dropout);
    config.text_vocab_size = json::optional_i64(root, "text_vocab_size", config.text_vocab_size);
    config.text_encoder_type = json::optional_string(root, "text_encoder_type", config.text_encoder_type);
    config.pretrained_projector_type =
        json::optional_string(root, "pretrained_projector_type", config.pretrained_projector_type);
    config.pretrained_projector_hidden_ratio =
        json::optional_f32(root, "pretrained_projector_hidden_ratio", config.pretrained_projector_hidden_ratio);
    config.text_add_bos = json::optional_bool(root, "text_add_bos", config.text_add_bos);
    config.text_dim = json::optional_i64(root, "text_dim", config.text_dim);
    config.text_layers = json::optional_i64(root, "text_layers", config.text_layers);
    config.text_heads = json::optional_i64(root, "text_heads", config.text_heads);
    config.use_caption_condition = json::optional_bool(root, "use_caption_condition", config.use_caption_condition);
    const bool legacy_speaker_default = !config.use_caption_condition;
    config.use_speaker_condition = json::optional_nullable_bool(root, "use_speaker_condition", legacy_speaker_default);
    config.caption_vocab_size = json::optional_nullable_i64(root, "caption_vocab_size", 0);
    config.caption_dim = json::optional_nullable_i64(root, "caption_dim", 0);
    config.caption_layers = json::optional_nullable_i64(root, "caption_layers", 0);
    config.caption_heads = json::optional_nullable_i64(root, "caption_heads", 0);
    config.caption_mlp_ratio = optional_ratio(root, "caption_mlp_ratio", 0.0F);
    config.speaker_dim = json::optional_i64(root, "speaker_dim", config.speaker_dim);
    config.speaker_layers = json::optional_i64(root, "speaker_layers", config.speaker_layers);
    config.speaker_heads = json::optional_i64(root, "speaker_heads", config.speaker_heads);
    config.speaker_patch_size = json::optional_i64(root, "speaker_patch_size", config.speaker_patch_size);
    config.timestep_embed_dim = json::optional_i64(root, "timestep_embed_dim", config.timestep_embed_dim);
    config.adaln_rank = json::optional_i64(root, "adaln_rank", config.adaln_rank);
    config.norm_eps = json::optional_f32(root, "norm_eps", config.norm_eps);
    config.use_duration_predictor = json::optional_bool(root, "use_duration_predictor", config.use_duration_predictor);
    config.duration_aux_dim = json::optional_i64(root, "duration_aux_dim", config.duration_aux_dim);
    config.duration_hidden_dim = json::optional_i64(root, "duration_hidden_dim", config.duration_hidden_dim);
    config.duration_layers = json::optional_i64(root, "duration_layers", config.duration_layers);
    config.duration_dropout = json::optional_f32(root, "duration_dropout", config.duration_dropout);
    config.duration_attention_heads =
        json::optional_i64(root, "duration_attention_heads", config.duration_attention_heads);
    config.duration_architecture =
        json::optional_string(root, "duration_architecture", config.duration_architecture);
    config.duration_token_init_frames =
        json::optional_f32(root, "duration_token_init_frames", config.duration_token_init_frames);
    config.duration_speaker_fusion =
        json::optional_string(root, "duration_speaker_fusion", config.duration_speaker_fusion);
    config.duration_caption_fusion =
        json::optional_nullable_string(root, "duration_caption_fusion", config.duration_caption_fusion);
    config.duration_caption_pooling =
        json::optional_nullable_string(root, "duration_caption_pooling", config.duration_caption_pooling);
    config.max_text_len = json::optional_i64(root, "max_text_len", config.max_text_len);
    config.max_caption_len = json::optional_i64(root, "max_caption_len", config.max_text_len);
    if (config.use_pretrained_text_encoder()) {
        if (!resources.has_file("pretrained_text_config")) {
            throw std::runtime_error("Irodori-TTS pretrained text encoder requires pretrained_text_config");
        }
        const auto encoder_config = resources.parse_json("pretrained_text_config");
        const auto model_type = json::optional_string(encoder_config, "model_type", "modernbert");
        config.use_t5gemma2 = model_type == "t5gemma2";
        if (!config.use_t5gemma2 && model_type != "modernbert") {
            throw std::runtime_error("Unsupported Irodori text encoder: " + model_type);
        }
        const auto & text_config = config.use_t5gemma2
            ? encoder_config.require("encoder").require("text_config") : encoder_config;
        config.pretrained_text_dim = json::require_i64(text_config, "hidden_size");
        config.pretrained_text_layers = json::require_i64(text_config, "num_hidden_layers");
        config.pretrained_text_heads = json::require_i64(text_config, "num_attention_heads");
        config.pretrained_text_intermediate_dim = json::require_i64(text_config, "intermediate_size");
        const int64_t local_attention = json::optional_i64(text_config, "local_attention", 0);
        config.pretrained_text_sliding_window = json::optional_i64(text_config, "sliding_window", local_attention / 2);
        if (!config.use_t5gemma2 && config.pretrained_text_sliding_window > 0) {
            ++config.pretrained_text_sliding_window;
        }
        if (const auto * rope = text_config.find("rope_parameters"); rope != nullptr && rope->is_object()) {
            if (const auto * full = rope->find("full_attention"); full != nullptr && full->is_object()) {
                config.pretrained_text_full_rope_theta =
                    json::optional_f32(*full, "rope_theta", config.pretrained_text_full_rope_theta);
            }
            if (const auto * sliding = rope->find("sliding_attention"); sliding != nullptr && sliding->is_object()) {
                config.pretrained_text_sliding_rope_theta =
                    json::optional_f32(*sliding, "rope_theta", config.pretrained_text_sliding_rope_theta);
            }
        }
        const auto & layer_types = text_config.require("layer_types").as_array();
        config.pretrained_text_layer_types.reserve(layer_types.size());
        for (const auto & value : layer_types) {
            const auto type = value.as_string();
            if (type != "full_attention" && type != "sliding_attention") {
                throw std::runtime_error("Unsupported Irodori attention layer type: " + type);
            }
            config.pretrained_text_layer_types.push_back(type);
        }
        if (config.use_t5gemma2) {
            auto & t5 = config.t5gemma2;
            t5.hidden_size = config.pretrained_text_dim;
            t5.layers = config.pretrained_text_layers;
            t5.attention_heads = config.pretrained_text_heads;
            t5.intermediate_size = config.pretrained_text_intermediate_dim;
            t5.vocab_size = json::require_i64(text_config, "vocab_size");
            if (t5.vocab_size != config.text_vocab_size) {
                throw std::runtime_error("Irodori T5Gemma2 vocabulary size mismatch");
            }
            t5.kv_heads = json::require_i64(text_config, "num_key_value_heads");
            t5.head_dim = json::require_i64(text_config, "head_dim");
            t5.attention_size = t5.attention_heads * t5.head_dim;
            t5.rms_norm_eps = json::require_f32(text_config, "rms_norm_eps");
            t5.query_pre_attn_scalar = json::require_f32(text_config, "query_pre_attn_scalar");
            t5.attn_logit_softcap = json::optional_nullable_f32(text_config, "attn_logit_softcapping", 0.0F);
            t5.use_qk_norm = true;
            config.t5gemma2_eoi_token_index = json::optional_i64(encoder_config, "eoi_token_index", 256000);
            if (json::optional_string(text_config, "hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh" ||
                json::optional_bool(text_config, "attention_bias", false)) {
                throw std::runtime_error("Unsupported Irodori T5Gemma2 activation or attention bias");
            }
            engine::io::require_positive(config.pretrained_text_sliding_window, "T5Gemma2 sliding_window");
            const auto & ropes = text_config.require("rope_parameters");
            for (const auto & type : config.pretrained_text_layer_types) {
                const auto & rope = ropes.require(type);
                t5.layer_rope_theta.push_back(json::require_f32(rope, "rope_theta"));
                const auto kind = json::optional_string(rope, "rope_type", "default");
                if (kind != "default" && kind != "linear") {
                    throw std::runtime_error("Unsupported Irodori T5Gemma2 RoPE: " + kind);
                }
                const float factor = kind == "linear" ? json::require_f32(rope, "factor") : 1.0F;
                if (!(factor > 0)) throw std::runtime_error("T5Gemma2 RoPE factor must be positive");
                t5.layer_rope_freq_scale.push_back(1.0F / factor);
            }
            // Validate dimensions before allocating model weights.
            std::vector<bool> sliding;
            for (const auto & type : config.pretrained_text_layer_types) sliding.push_back(type == "sliding_attention");
            modules::T5Gemma2EncoderModuleExp(t5, sliding, config.t5gemma2_eoi_token_index);
        }
    }

    engine::io::require_positive(config.latent_dim, "latent_dim");
    engine::io::require_positive(config.latent_patch_size, "latent_patch_size");
    engine::io::require_positive(config.model_dim, "model_dim");
    engine::io::require_positive(config.num_layers, "num_layers");
    engine::io::require_positive(config.num_heads, "num_heads");
    if (!(config.mlp_ratio > 0.0)) {
        throw std::runtime_error("mlp_ratio must be positive");
    }
    engine::io::require_positive(config.text_vocab_size, "text_vocab_size");
    engine::io::require_positive(config.text_dim, "text_dim");
    if (config.text_encoder_type != "scratch" && config.text_encoder_type != "pretrained") {
        throw std::runtime_error("Irodori-TTS text_encoder_type must be scratch or pretrained");
    }
    if (config.use_pretrained_text_encoder()) {
        engine::io::require_positive(config.pretrained_text_dim, "pretrained_text_dim");
        engine::io::require_positive(config.pretrained_text_layers, "pretrained_text_layers");
        engine::io::require_positive(config.pretrained_text_heads, "pretrained_text_heads");
        engine::io::require_positive(config.pretrained_text_intermediate_dim, "pretrained_text_intermediate_dim");
        if (!config.use_t5gemma2) engine::io::require_divisible(
            config.pretrained_text_dim,
            config.pretrained_text_heads,
            "pretrained_text_dim / pretrained_text_heads");
        if (static_cast<int64_t>(config.pretrained_text_layer_types.size()) != config.pretrained_text_layers) {
            throw std::runtime_error("Irodori-TTS pretrained text layer_types count mismatch");
        }
        if (config.pretrained_projector_type != "linear" &&
            config.pretrained_projector_type != "residual_mlp") {
            throw std::runtime_error("Irodori-TTS pretrained_projector_type must be linear or residual_mlp");
        }
        engine::io::require_positive(config.pretrained_projector_hidden_ratio, "pretrained_projector_hidden_ratio");
    } else {
        engine::io::require_positive(config.text_layers, "text_layers");
        engine::io::require_positive(config.text_heads, "text_heads");
        engine::io::require_divisible(config.text_dim, config.text_heads, "text_dim / text_heads");
    }
    engine::io::require_positive(config.timestep_embed_dim, "timestep_embed_dim");
    engine::io::require_positive(config.adaln_rank, "adaln_rank");
    engine::io::require_positive(config.norm_eps, "norm_eps");
    engine::io::require_divisible(config.model_dim, config.num_heads, "model_dim / num_heads");
    if (config.use_speaker_condition) {
        engine::io::require_positive(config.speaker_dim, "speaker_dim");
        engine::io::require_positive(config.speaker_layers, "speaker_layers");
        engine::io::require_positive(config.speaker_heads, "speaker_heads");
        engine::io::require_positive(config.speaker_patch_size, "speaker_patch_size");
        engine::io::require_divisible(config.speaker_dim, config.speaker_heads, "speaker_dim / speaker_heads");
    }
    if (config.use_caption_condition) {
        engine::io::require_positive(config.caption_vocab_size_resolved(), "caption_vocab_size");
        engine::io::require_positive(config.caption_dim_resolved(), "caption_dim");
        if (!config.use_pretrained_text_encoder()) {
            engine::io::require_positive(config.caption_layers_resolved(), "caption_layers");
            engine::io::require_positive(config.caption_heads_resolved(), "caption_heads");
            engine::io::require_divisible(config.caption_dim_resolved(), config.caption_heads_resolved(), "caption_dim / caption_heads");
        }
    }
    if (config.use_duration_predictor) {
        engine::io::require_positive(config.duration_aux_dim, "duration_aux_dim");
        engine::io::require_positive(config.duration_hidden_dim, "duration_hidden_dim");
        engine::io::require_positive(config.duration_layers, "duration_layers");
    }
    return config;
}

IrodoriModelConfig parse_config(const assets::ResourceBundle & resources) {
    return parse_model_config(resources);
}

void validate_model_weights(const IrodoriModelConfig & config, const assets::TensorSource & source) {
    if (config.use_pretrained_text_encoder()) {
        if (config.use_t5gemma2) {
            assets::require_tensor_shape(source, "pretrained_text_backbone.backbone.embed_tokens.weight",
                                         {config.text_vocab_size, config.pretrained_text_dim});
            assets::require_tensor_shape(source, "pretrained_text_backbone.backbone.embed_tokens.eoi_embedding",
                                         {config.pretrained_text_dim});
            assets::require_tensor_shape(source, "pretrained_text_backbone.backbone.norm.weight",
                                         {config.pretrained_text_dim});
        } else {
            assets::require_tensor_shape(
                source,
                "pretrained_text_backbone.backbone.embeddings.tok_embeddings.weight",
                {config.text_vocab_size, config.pretrained_text_dim});
            assets::require_tensor_shape(
                source,
                "pretrained_text_backbone.backbone.embeddings.norm.weight",
                {config.pretrained_text_dim});
            assets::require_tensor_shape(
                source,
                "pretrained_text_backbone.backbone.final_norm.weight",
                {config.pretrained_text_dim});
            assets::require_tensor_shape(
                source,
                "pretrained_text_backbone.backbone.layers.0.attn.Wqkv.weight",
                {3 * config.pretrained_text_dim, config.pretrained_text_dim});
            assets::require_tensor_shape(
                source,
                "pretrained_text_backbone.backbone.layers.0.mlp.Wi.weight",
                {2 * config.pretrained_text_intermediate_dim, config.pretrained_text_dim});
        }
        assets::require_tensor_shape(source, "text_encoder.projector.weight", {config.text_dim, config.pretrained_text_dim});
        assets::require_tensor_shape(source, "text_norm.weight", {config.text_dim});
        if (config.pretrained_projector_type == "residual_mlp") {
            const int64_t hidden = static_cast<int64_t>(
                std::round(static_cast<double>(config.text_dim) * config.pretrained_projector_hidden_ratio));
            assets::require_tensor_shape(source, "text_encoder.residual_norm.weight", {config.pretrained_text_dim});
            assets::require_tensor_shape(source, "text_encoder.residual_up.weight", {hidden, config.pretrained_text_dim});
            assets::require_tensor_shape(source, "text_encoder.residual_down.weight", {config.text_dim, hidden});
        }
    } else {
        assets::require_tensor_shape(source, "text_encoder.text_embedding.weight", {config.text_vocab_size, config.text_dim});
        assets::require_tensor_shape(source, "text_norm.weight", {config.text_dim});
    }
    assets::require_tensor_shape(source, "in_proj.weight", {config.model_dim, config.patched_latent_dim()});
    assets::require_tensor_shape(source, "in_proj.bias", {config.model_dim});
    assets::require_tensor_shape(source, "out_norm.weight", {config.model_dim});
    assets::require_tensor_shape(source, "out_proj.weight", {config.patched_latent_dim(), config.model_dim});
    assets::require_tensor_shape(source, "out_proj.bias", {config.patched_latent_dim()});
    assets::require_tensor_shape(source, "blocks.0.attention.wq.weight", {config.model_dim, config.model_dim});
    assets::require_tensor_shape(source, "blocks.0.attention.wk_text.weight", {config.model_dim, config.text_dim});
    if (config.use_speaker_condition) {
        assets::require_tensor_shape(source, "speaker_encoder.in_proj.weight", {config.speaker_dim, config.speaker_patched_latent_dim()});
        assets::require_tensor_shape(source, "speaker_norm.weight", {config.speaker_dim});
        const int64_t speaker_ffn = static_cast<int64_t>(config.speaker_dim * config.speaker_mlp_ratio);
        for (int64_t layer = 0; layer < config.speaker_layers; ++layer) {
            const std::string prefix = "speaker_encoder.blocks." + std::to_string(layer) + ".mlp.";
            assets::require_tensor_shape(source, prefix + "w1.weight", {speaker_ffn, config.speaker_dim});
            assets::require_tensor_shape(source, prefix + "w2.weight", {config.speaker_dim, speaker_ffn});
            assets::require_tensor_shape(source, prefix + "w3.weight", {speaker_ffn, config.speaker_dim});
        }
        assets::require_tensor_shape(source, "blocks.0.attention.wk_speaker.weight", {config.model_dim, config.speaker_dim});
    }
    if (config.use_caption_condition) {
        if (config.use_pretrained_text_encoder()) {
            assets::require_tensor_shape(source, "caption_encoder.projector.weight", {config.caption_dim_resolved(), config.pretrained_text_dim});
            assets::require_tensor_shape(source, "caption_norm.weight", {config.caption_dim_resolved()});
            if (config.pretrained_projector_type == "residual_mlp") {
                const int64_t hidden = static_cast<int64_t>(
                    std::round(static_cast<double>(config.caption_dim_resolved()) *
                               config.pretrained_projector_hidden_ratio));
                assets::require_tensor_shape(source, "caption_encoder.residual_norm.weight", {config.pretrained_text_dim});
                assets::require_tensor_shape(source, "caption_encoder.residual_up.weight", {hidden, config.pretrained_text_dim});
                assets::require_tensor_shape(source, "caption_encoder.residual_down.weight", {config.caption_dim_resolved(), hidden});
            }
        } else {
            assets::require_tensor_shape(source, "caption_encoder.text_embedding.weight", {config.caption_vocab_size_resolved(), config.caption_dim_resolved()});
            assets::require_tensor_shape(source, "caption_norm.weight", {config.caption_dim_resolved()});
        }
        assets::require_tensor_shape(source, "blocks.0.attention.wk_caption.weight", {config.model_dim, config.caption_dim_resolved()});
        assets::require_tensor_shape(source, "blocks.0.attention.wv_caption.weight", {config.model_dim, config.caption_dim_resolved()});
        assets::require_tensor_shape(source, "duration_predictor.null_caption", {config.caption_dim_resolved()});
        assets::require_tensor_shape(
            source,
            "duration_predictor.token_blocks.0.caption_modulation.weight",
            {3 * config.duration_hidden_dim, config.caption_dim_resolved()});
    }
    if (config.use_duration_predictor) {
        assets::require_tensor_shape(source, "duration_predictor.token_input_proj.weight", {config.duration_hidden_dim, config.text_dim});
        assets::require_tensor_shape(source, "duration_predictor.token_out_proj.weight", {1, config.duration_hidden_dim});
        assets::require_tensor_shape(source, "duration_predictor.token_out_proj.bias", {1});
    }
}

void validate_codec_weights(const IrodoriCodecConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "quantizer.out_proj.weight_v", {config.latent_dim, config.codebook_dim, 1});
    assets::require_tensor_shape(source, "quantizer.out_proj.weight_g", {config.latent_dim, 1, 1});
    assets::require_tensor_shape(source, "quantizer.out_proj.bias", {config.latent_dim});
    assets::require_tensor_shape(source, "decoder.model.0.weight_v", {config.decoder_dim, config.latent_dim, 7});
    assets::require_tensor_shape(source, "decoder.model.0.weight_g", {config.decoder_dim, 1, 1});
    assets::require_tensor_shape(source, "decoder.model.0.bias", {config.decoder_dim});
}

void validate_weight_anchors(const IrodoriTTSAssets & assets) {
    validate_model_weights(assets.config, *assets.model_weights);
    validate_codec_weights(assets.codec, *assets.codec_weights);
}

}  // namespace

int64_t IrodoriModelConfig::patched_latent_dim() const noexcept {
    return latent_dim * latent_patch_size;
}

int64_t IrodoriModelConfig::speaker_patched_latent_dim() const noexcept {
    return patched_latent_dim() * speaker_patch_size;
}

bool IrodoriModelConfig::use_pretrained_text_encoder() const noexcept {
    return text_encoder_type == "pretrained";
}

int64_t IrodoriModelConfig::caption_vocab_size_resolved() const noexcept {
    return caption_vocab_size > 0 ? caption_vocab_size : text_vocab_size;
}

int64_t IrodoriModelConfig::caption_dim_resolved() const noexcept {
    return caption_dim > 0 ? caption_dim : text_dim;
}

int64_t IrodoriModelConfig::caption_layers_resolved() const noexcept {
    return caption_layers > 0 ? caption_layers : text_layers;
}

int64_t IrodoriModelConfig::caption_heads_resolved() const noexcept {
    return caption_heads > 0 ? caption_heads : text_heads;
}

double IrodoriModelConfig::caption_mlp_ratio_resolved() const noexcept {
    return caption_mlp_ratio > 0.0F ? caption_mlp_ratio : text_mlp_ratio;
}

std::shared_ptr<const IrodoriTTSAssets> load_irodori_tts_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<IrodoriTTSAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("irodori_tts"));

    assets->config = parse_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("model_weights");
    assets->codec_weights = assets->resources.open_tensor_source("codec_weights");
    validate_weight_anchors(*assets);
    return assets;
}

}  // namespace engine::models::irodori_tts
