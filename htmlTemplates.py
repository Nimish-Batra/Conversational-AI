css='''
<style>
.chat-message {padding:1.5rem; border-redius:0.5rem; margin-bottom:1rem; display:flex}
.chat-message.user {background-color: #EEEEEE}
.chat-message.bot {background-color: #EEEEEE}
.message {color: #000000;}
.chat-message.avtar {width: 15%; color: #ffffff;}
.chat-message.avtar img {
max-width:78px;
max-height: 78px;
border-radius: 50%;
object-fit: cover;
}
.chat-message.message {with: 85%;
padding: 0 1.5rem;
color: #fff;}
'''

bot_template ='''
<div class="chat-message bot">
     <div class="message"> {{MSG}} <br><br> Source:<a>{{source_url}}</a> page: {{page_number}}</div
</div>
'''

user_template='''
<div class="chat-message bot">
    <div class="message"> {{MSG}}</div>
</div>
'''