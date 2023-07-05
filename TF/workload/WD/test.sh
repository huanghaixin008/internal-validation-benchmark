saved_model_cli run --dir ./wide_deep_saved_models/ \
    --tag_set serve --signature_def="predict" \
    --input_examples='examples=[{"age":[46.], "education_num":[10.], "capital_gain":[7688.], "capital_loss":[0.], "hours_per_week":[38.]}, {"age":[24.], "education_num":[13.], "capital_gain":[0.], "capital_loss":[0.], "hours_per_week":[50.]}]'
