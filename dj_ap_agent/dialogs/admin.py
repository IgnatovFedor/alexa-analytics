from django.contrib import admin

# Register your models here.
from django.contrib import admin
from dialogs.models import Dialog, Author, Utterance, Annotation, UtteranceHypothesis

class AuthorAdmin(admin.ModelAdmin):
    pass
admin.site.register(Author, AuthorAdmin)


class UtteranceInline(admin.TabularInline):
    model = Utterance
    # raw_id_fields = ("parent_utterance",)
    # extra = 0

class DialogAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'start_time', 'rating')
    # inlines = [
    #     UtteranceInline,
    # ]
    change_form_template = 'admin/dialogs/dialog_change_form.html'
    fields = ('conversation_id', 'dp_id', 'start_time', 'rating', 'human', 'bot', 'view_dialog')
    readonly_fields = ('conversation_id', 'dp_id', 'start_time', 'rating', 'human', 'bot', 'view_dialog')
    search_fields = ['dp_id', 'conversation_id']
    list_filter = ('rating',)
    ordering = ('-start_time',)

    def view_dialog(self, obj):
        return str(obj)

    view_dialog.empty_value_display = '???'
    
    def get_search_results(self, request, queryset, search_term):
        queryset, use_distinct = super().get_search_results(request, queryset, search_term)
        if search_term:
            try:
                utts = Utterance.objects.filter(text__contains=search_term).select_related("parent_dialog")
                dialog_ids = [utt.parent_dialog.id for utt in utts]
                unique_d_ids = list(set(dialog_ids))
            except ValueError:
                pass
            else:
                queryset |= self.model.objects.filter(id__in=unique_d_ids)
        return queryset, use_distinct

admin.site.register(Dialog, DialogAdmin)

class DialogInline(admin.TabularInline):
    model = Dialog
#     # raw_id_fields = ("parent_utterance",)
#     # extra = 0


class AnnotationInline(admin.TabularInline):
    model = Annotation
    raw_id_fields = ("parent_utterance",)
    extra = 0


class UtteranceHypothesesInline(admin.TabularInline):
    model = UtteranceHypothesis
    raw_id_fields = ("parent_utterance",)
    readonly_fields = ("parent_utterance",)
    extra = 0
    inlines = [
        UtteranceInline,
    ]

class UtteranceAdmin(admin.ModelAdmin):
    inlines = [
        AnnotationInline,
        UtteranceHypothesesInline
    ]
    raw_id_fields = ('parent_dialog', 'author')
    readonly_fields = ("parent_dialog", "author")
    list_display = ('text', 'author', 'timestamp')

admin.site.register(Utterance, UtteranceAdmin)


class UtteranceHypothesisAdmin(admin.ModelAdmin):
    readonly_fields = ('parent_utterance', 'text', 'skill_name', 'confidence', 'other_attrs')
admin.site.register(UtteranceHypothesis, UtteranceHypothesisAdmin)

class AnnotationAdmin(admin.ModelAdmin):
    readonly_fields = ('parent_utterance', 'annotation_type', 'annotation_dict')
admin.site.register(Annotation, AnnotationAdmin)
