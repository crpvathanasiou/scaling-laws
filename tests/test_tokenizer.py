import json

from scaling_laws.tokenizer.train_tokenizer import (
    SPECIAL_TOKENS,
    add_special_token_postprocessor,
    build_tokenizer,
    clean_text,
    estimate_embedding_share,
    train_final_tokenizer,
)


def test_clean_text():
    assert clean_text("  γεια σου  ") == "γεια σου"
    assert clean_text("") is None
    assert clean_text("   ") is None
    assert clean_text(None) is None
    assert clean_text(123) is None


def test_estimate_embedding_share_tied_and_untied():
    tied = estimate_embedding_share(
        vocab_size=8000,
        d_model=128,
        total_params=2_000_000,
        tied_embeddings=True,
    )
    untied = estimate_embedding_share(
        vocab_size=8000,
        d_model=128,
        total_params=2_000_000,
        tied_embeddings=False,
    )

    assert tied == 8000 * 128 / 2_000_000
    assert untied == 2 * 8000 * 128 / 2_000_000
    assert untied == 2 * tied


def test_postprocessor_adds_bos_and_eos_tokens():
    texts = [
        "Μια φορά κι έναν καιρό",
        "Το παιδί έτρεξε στο σπίτι",
        "Η γάτα κοιμόταν στο παράθυρο",
    ]

    tokenizer, trainer = build_tokenizer(vocab_size=100, min_frequency=1)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer = add_special_token_postprocessor(tokenizer)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    encoded = tokenizer.encode("Μια μικρή ιστορία")

    assert encoded.ids[0] == bos_id
    assert encoded.ids[-1] == eos_id


def test_train_final_tokenizer_saves_expected_files(tmp_path):
    train_texts = [
        "Μια φορά κι έναν καιρό ήταν ένα μικρό παιδί.",
        "Το παιδί αγαπούσε τα βιβλία και τις ιστορίες.",
        "Κάθε βράδυ διάβαζε ένα νέο παραμύθι.",
        "Η γάτα κοιμόταν δίπλα στο τζάκι.",
    ]

    output_dir = tmp_path / "final_vocab_128"

    tokenizer = train_final_tokenizer(
        train_texts=train_texts,
        vocab_size=128,
        min_frequency=1,
        output_dir=str(output_dir),
        metadata={"selected_vocab_size": 128},
    )

    assert tokenizer.get_vocab_size() <= 128

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer_config = output_dir / "tokenizer_config.json"
    vocab_txt = output_dir / "vocab.txt"
    special_tokens_map = output_dir / "special_tokens_map.json"

    assert tokenizer_json.exists()
    assert tokenizer_config.exists()
    assert vocab_txt.exists()
    assert special_tokens_map.exists()

    config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    assert config["model"] == "BPE"
    assert config["normalization"] == "NFKC"
    assert config["pre_tokenizer"] == "Whitespace"
    assert config["special_tokens"] == SPECIAL_TOKENS

    special_map = json.loads(special_tokens_map.read_text(encoding="utf-8"))
    assert special_map["pad_token"] == "[PAD]"
    assert special_map["unk_token"] == "[UNK]"
    assert special_map["bos_token"] == "[BOS]"
    assert special_map["eos_token"] == "[EOS]"
    assert isinstance(special_map["pad_token_id"], int)
    assert isinstance(special_map["eos_token_id"], int)
